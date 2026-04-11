from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import random


def glorot_init(key, in_dim: int, out_dim: int, *, dtype=jnp.float64):
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    return {
        "W": random.uniform(key, (int(in_dim), int(out_dim)), minval=-limit, maxval=limit, dtype=dtype),
        "b": jnp.zeros((int(out_dim),), dtype=dtype),
    }


def init_mlp_params(key, layer_sizes: list[int] | tuple[int, ...], *, dtype=jnp.float64):
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain at least input and output dimensions.")
    keys = random.split(key, len(layer_sizes) - 1)
    params = []
    for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        params.append(glorot_init(k, int(din), int(dout), dtype=dtype))
    return params


def mlp_apply(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


def make_batched_mlp_apply():
    return jax.jit(jax.vmap(mlp_apply, in_axes=(None, 0)))
