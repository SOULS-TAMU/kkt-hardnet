# Development version notes

This development branch retains public benchmark APIs while using one OHAGI
learning/inference engine and PURIN projection implementation.

- Frontends supply problem definitions, task policies and reporting.
- DAEHN additionally processes differential terms and reconstructs states.
- Functional JAX and optional Flax backbones use the shared engine.
- Optional Orbax checkpoints include parameters and optimizer state.
- Full-model saves and deployment use shared OHAGI JAX formats.
- Legacy model readers and native export backends are removed.

## Source organization

```text
src/<frontend>/
    __init__.py
    api/             # Frontend problem definitions and policies
    serialization/   # Adapters to OHAGI model resources
```

DAEHN keeps its mathematical processing in `api/differential`. KKTHN also retains
its public import aliases. OHAGI and PURIN do not import frontend implementations.

## Compatibility policy

The [validation policy](../.github/validation/README.md) describes API snapshots,
numerical tolerances, exact checkpoint-resume checks and quality gates.
See the [changelog](../CHANGELOG.md) for history. Unreleased development changes
should not be assumed to exist in published packages.

[Back to README](../README.md)
