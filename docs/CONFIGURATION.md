# Configuration

## Training and loss

The `train` mapping controls a run. Use an explicit backbone for a custom network;
benchmark-compatible aliases need not appear in a new example.

| Setting | Role |
|---|---|
| `epochs` | Requested total training horizon |
| `batch_size` | Fixed training/projection batch shape |
| `learning_rate`, `seed` | Optimizer step size and initialization |
| `train_frac` | Observation split fraction |
| `dtype` | Precision, for example `float64` |
| `epoch_mlp`, `pretrain` | Raw-network pretraining/phase policy |
| `output_dir`, `print_every` | Run location and logging cadence |

`Loss.mse()` selects regression. Optimization uses a symbolic objective; inverse
estimation introduces unknown coefficients. Consistency weights affect training,
not the definition of a physical constraint. The [API](api/index.rst) records
constructor signatures.

## Projection and architecture

Choose a method on `Projection(method=...)` with compatible configuration. The
standard distance objective uses identity metric. Select a solver matching the
actual affine/nonlinear constraint structure; see
[PURIN](https://github.com/SOULS-TAMU/Purin) for numerical requirements.

`MLP(hidden_sizes=(32, 32))` is functional JAX.
`MLP(hidden_sizes=(32, 32), backend="flax")` uses the optional Flax adapter.
Keep architecture, batching, data order and phase settings when resuming training.
