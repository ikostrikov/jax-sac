import enum
import typing

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.lax_numpy import square, squeeze

from jax_rl.networks.common import Params, PRNGKey

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

import enum
import typing

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


class MaskType(enum.Enum):
    input = 1
    hidden = 2
    output = 3


@jax.util.cache()
def get_mask(input_dim: int, output_dim: int, randvar_dim: int,
             mask_type: MaskType) -> jnp.DeviceArray:
    """
    Create a mask for MADE.

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf

    Args:
        input_dim: Dimensionality of the inputs.
        output_dim: Dimensionality of the outputs.
        rand_var_dim: Dimensionality of the random variable.
        mask_type: MaskType.

    Returns:
        A mask.
    """
    if mask_type == MaskType.input:
        in_degrees = jnp.arange(input_dim) % randvar_dim
    else:
        in_degrees = jnp.arange(input_dim) % (randvar_dim - 1)

    if mask_type == MaskType.output:
        out_degrees = jnp.arange(output_dim) % randvar_dim - 1
    else:
        out_degrees = jnp.arange(output_dim) % (randvar_dim - 1)

    in_degrees = jnp.expand_dims(in_degrees, 0)
    out_degrees = jnp.expand_dims(out_degrees, -1)
    return (out_degrees >= in_degrees).astype(jnp.float32).transpose()


class MaskedDense(nn.Dense):
    event_size: int = 1
    mask_type: MaskType = MaskType.hidden
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel', self.kernel_init,
                            (inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)

        mask = get_mask(*kernel.shape, self.event_size, self.mask_type)
        kernel = kernel * mask

        y = jax.lax.dot_general(inputs,
                                kernel,
                                (((inputs.ndim - 1, ), (0, )), ((), ())),
                                precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features, ))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


class MaskedMLP(nn.Module):
    features: typing.Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, inputs, conds):
        x = inputs
        x_conds = conds
        for i, feat in enumerate(self.features):
            if i == 0:
                mask_type = MaskType.input
            elif i + 1 < len(self.features):
                mask_type = MaskType.hidden
            else:
                mask_type = MaskType.output
            x = MaskedDense(feat,
                            event_size=inputs.shape[-1],
                            mask_type=mask_type)(x)
            x_conds = nn.Dense(feat)(x_conds)
            x = x + x_conds
            if i + 1 < len(self.features) or self.activate_final:
                x = nn.relu(x)
                x_conds = nn.relu(x_conds)
        return x


class MADETanhMixturePolicy(nn.Module):
    features: typing.Sequence[int]
    num_components: int = 10

    @nn.compact
    def __call__(self, states, actions, temperature: float = 1.0):
        outputs = MaskedMLP(
            (*self.features,
             3 * self.num_components * actions.shape[-1]))(actions, states)
        means, log_scales, logits = jnp.split(outputs, 3, axis=-1)
        means = means + self.param('means', nn.initializers.normal(1.0),
                                   (means.shape[-1], ))

        log_scales = jnp.clip(log_scales, LOG_STD_MIN, LOG_STD_MAX)

        def reshape(x):
            if len(x.shape) == 1:
                x = jnp.reshape(x, [self.num_components, actions.shape[-1]])
                return jnp.transpose(x, (1, 0))
            else:
                x = jnp.reshape(x,
                                [-1, self.num_components, actions.shape[-1]])
                return jnp.transpose(x, (0, 2, 1))

        means = reshape(means)
        log_scales = reshape(log_scales)
        logits = reshape(logits)

        dist = distrax.Normal(loc=means,
                              scale=jnp.exp(log_scales) * temperature)

        dist = distrax.MixtureSameFamily(distrax.Categorical(logits=logits),
                                         dist)

        dist = distrax.Independent(dist, reinterpreted_batch_ndims=1)
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), 1))


@jax.partial(jax.jit, static_argnums=(1, 4, 5))
def sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        action_dim: int,
        temperature: float = 1.0) -> typing.Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, action_dim)

    def f(i, x):
        dist = actor_def.apply({'params': actor_params}, observations, x,
                               temperature)
        samples = dist.sample(seed=keys[i])
        return jax.ops.index_update(x, jax.ops.index[..., i], samples[..., i])

    sampled_actions = jax.lax.fori_loop(
        0, action_dim, f,
        jnp.zeros((*observations.shape[:-1], action_dim), jnp.float32))

    return rng, sampled_actions
