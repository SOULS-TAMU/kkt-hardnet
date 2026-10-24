"""Internal tree and serialization helpers."""

from __future__ import annotations
import jax
import jax.numpy as jnp
from ohagi._utils.serialization import json_safe as _json_safe


def _copy_tree(tree):
    return jax.tree_util.tree_map(lambda value: jnp.array(value), tree)


__all__ = ["_copy_tree", "_json_safe"]
