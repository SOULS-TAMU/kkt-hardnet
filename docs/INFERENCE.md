# Export and repeated inference

After training, call `model.export_inference("deployment/metadata.json")`.
Full-model reload uses `KKTHardNet().load(result["metadata_path"])`; use the
shared loader below for a deployment-only bundle.

## Load once and reuse

```python
from ohagi.inference import load

predictor = load("deployment/metadata.json")
for value in [0.4, 0.7, 1.0]:
    prediction = predictor.predict([value])
```

The loader prepares and warms the exported signature. Reuse the predictor across
calls. `predict` synchronizes to NumPy; `predict_jax` returns a JAX array and may
execute asynchronously. Synchronize results when timing. Export the shapes and
dtype required by the application.

## Files to retain

| Artifact | Purpose |
|---|---|
| Full-model metadata plus referenced weights/formulation | Frontend model reload |
| Deployment metadata plus referenced `.jax_export` graph | Repeated prediction |
| Orbax checkpoint directory | Training continuation with optimizer state |

Pass only the metadata path to the loader, but move its referenced files together.
Metadata alone is not the complete model. Exported deployment needs compatible
OHAGI/JAX dependencies; it does not need frontend factories, Flax or Orbax.

Full models use `ohagi_model_v1` and `ohagi_pytree_npz_v1`; deployment uses
`ohagi_jax_inference_v1` and `ohagi_jax_program_v1`. Only the shared JAX pathway
is supported. JAX export uses StableHLO internally, without a separate native
compiler or standalone StableHLO deployment option.

[Back to README](../README.md)
