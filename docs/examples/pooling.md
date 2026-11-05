# Pooling problem

This benchmark learns a constrained mapping with four input parameters (`x1` through `x4`) and five outputs (`y1` through `y5`).
The included CSV is `Pooling_dataset.csv`; its headers match the names declared by the
training problem. Constraints are supplied by the case's symbolic formulation
and enforced through its prepared PURIN distance projection.

## Train and analyze

Clone [kkthn-benchmarks](https://github.com/RG8420/kkthn-benchmarks) and follow
its installation instructions. From that repository's root:

```shell
python run.py pooling train --epochs 2
```

Omit the short-run override for the configured experiment. New outputs are placed
under `results/pooling/`; the bundled trained snapshots remain in `pooling/models/`.

The [case guide](https://github.com/RG8420/kkthn-benchmarks/tree/main/pooling)
documents the dataset, full constraint declarations, default training setup and
inference commands. For a newly trained model, select its emitted metadata:

```shell
python run.py pooling infer --metadata results/pooling/model/new_runs/<run>/metadata.json
```

`<run>` means the actual run path printed by training. Included historical
checkpoints require the matching runtime described in the benchmark README;
they were not converted to a new format. Report prediction error and physical
constraint violation separately, and distinguish compilation from warmed timing.
