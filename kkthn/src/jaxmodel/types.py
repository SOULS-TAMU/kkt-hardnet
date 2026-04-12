from typing import Callable, Dict, Tuple, Any
import jax.numpy as jnp

Array = jnp.ndarray
ParamsDict = Dict[str, Array]
VarsDict = Dict[str, Array]

ScalarModelFun = Callable[[ParamsDict, VarsDict], Array]
VectorModelFun = Callable[[ParamsDict, VarsDict], Array]
FlatScalarFun = Callable[[ParamsDict, Array], Array]
FlatVectorFun = Callable[[ParamsDict, Array], Array]