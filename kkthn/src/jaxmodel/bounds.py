from dataclasses import dataclass
from typing import Optional, Callable
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class BoundSpec:
    lower_fun: Optional[Callable[[dict], Array]] = None
    upper_fun: Optional[Callable[[dict], Array]] = None