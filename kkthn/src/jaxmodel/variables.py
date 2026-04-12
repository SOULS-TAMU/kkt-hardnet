from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class VariableSpec:
    names: List[str]
    shapes: Dict[str, Tuple[int, ...]]
    sizes: Dict[str, int]
    slices: Dict[str, slice]
    total_size: int

    def pack(self, values: Dict[str, Array], dtype=jnp.float64) -> Array:
        parts = []
        for name in self.names:
            if name not in values:
                raise KeyError(f"Missing variable block '{name}'")
            arr = jnp.asarray(values[name], dtype=dtype).reshape(-1)
            if arr.size != self.sizes[name]:
                raise ValueError(
                    f"Variable '{name}' expected size {self.sizes[name]}, got {arr.size}"
                )
            parts.append(arr)
        if len(parts) == 0:
            return jnp.zeros((0,), dtype=dtype)
        return jnp.concatenate(parts, axis=0)

    def unpack(self, y: Array) -> Dict[str, Array]:
        if y.ndim != 1:
            raise ValueError(f"Expected flat vector, got shape {y.shape}")
        if y.shape[0] != self.total_size:
            raise ValueError(f"Expected vector length {self.total_size}, got {y.shape[0]}")

        out = {}
        for name in self.names:
            sl = self.slices[name]
            block = y[sl]
            shape = self.shapes[name]
            if shape == (1,):
                out[name] = block[0]
            else:
                out[name] = block.reshape(shape)
        return out


class VariableBuilder:
    def __init__(self):
        self._names: List[str] = []
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._sizes: Dict[str, int] = {}

    def add_scalar(self, name: str) -> "VariableBuilder":
        return self.add_block(name, (1,))

    def add_vector(self, name: str, n: int) -> "VariableBuilder":
        return self.add_block(name, (n,))

    def add_matrix(self, name: str, shape: Tuple[int, int]) -> "VariableBuilder":
        return self.add_block(name, shape)

    def add_tensor(self, name: str, shape: Tuple[int, ...]) -> "VariableBuilder":
        return self.add_block(name, shape)

    def add_block(self, name: str, shape: Union[int, Tuple[int, ...]]) -> "VariableBuilder":
        if isinstance(shape, int):
            shape = (shape,)
        if name in self._shapes:
            raise ValueError(f"Variable '{name}' already exists")

        size = 1
        for s in shape:
            size *= s

        self._names.append(name)
        self._shapes[name] = tuple(shape)
        self._sizes[name] = size
        return self

    def build(self) -> VariableSpec:
        start = 0
        slices = {}
        for name in self._names:
            size = self._sizes[name]
            slices[name] = slice(start, start + size)
            start += size

        return VariableSpec(
            names=list(self._names),
            shapes=dict(self._shapes),
            sizes=dict(self._sizes),
            slices=slices,
            total_size=start,
        )