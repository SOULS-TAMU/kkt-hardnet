# Compatibility and numerical checks

## Python and platforms

The packages require **Python 3.10 or newer**. GitHub CI tests its declared
Python/JAX combinations on Linux, Windows and macOS, using CPU as the baseline.
A Python version satisfying the lower bound still needs compatible JAX and
optional-dependency wheels. Future Python releases are not automatically certified.

Start on CPU. GPU installation requires a supported JAX/CUDA combination,
hardware and drivers; CPU CI does not certify every GPU setup.

## Precision, convergence and timing

Enable float64 before creating arrays when using tight constraint tolerances:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

A tolerance is not a promise of convergence for every problem. Check feasibility,
iteration limits and gradient eligibility. Physical constraint violation and
scaled solver/KKT residuals are different quantities.

Loading and compilation are startup costs. Time repeated inference after warmup
with unchanged shape/dtype, synchronizing device outputs with
`jax.block_until_ready`. Report loading, warmup and steady-state inference separately.

## Reproducibility

Record package/JAX versions, precision, device, seeds, data splits, architecture
and solver settings. Compare predictions, physical residuals and gradients with
appropriate tolerances. Wall-clock timings depend on hardware and execution context.
