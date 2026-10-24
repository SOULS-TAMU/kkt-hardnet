# Contributing

Keep numerical changes separate from API and packaging changes. Explain the
problem, resulting behavior, compatibility implications, and validation in a
pull request. Use the shared OHAGI training and deployment services and the
canonical PURIN solvers rather than adding frontend numerical copies.

For development, install the five local packages in editable mode in dependency
order, using `python -m pip install -e ./purin -e ./ohagi -e ./kkthn -e ./daehn
-e ./nlpoptnet` from the stack root. Optional OHAGI extras are `flax`, `orbax`,
`optax`, and `plotting`.

Before changing solver behavior, compare outputs, parameter gradients, and
constraint residuals with the previous version under identical JAX, device,
precision, and random-seed settings. Training changes should compare resumed
and uninterrupted runs. Deployment changes should exercise export, reload,
and repeated vector inference in a fresh process. Keep benchmark call
signatures and historical aliases compatible.

Build distributions with `python -m build` inside each library directory and
check them with `python -m twine check dist/*`. Install the wheels into a
separate environment and verify imports and deployment. Do not commit
environments, generated results, checkpoints, caches, or build artifacts.


## Readable code and API documentation

Format library sources with `python -m ruff format src` and check them with
`python -m ruff format --check src`. The package configuration uses 88 columns
and Python 3.10 syntax. Do not format benchmark scripts as part of library edits.

Use Google-style docstrings for new public APIs: a direct summary followed by
`Args:`, `Returns:` (or `Yields:`), `Raises:` and `Notes:` where they explain a
nontrivial contract. Preserve useful existing reference documentation. Simple
properties and protocol adapters may use one precise summary sentence.

For numerical interfaces, state array shapes, axis order, physical versus
normalized coordinates, residual sign conventions, masks, warm starts and
gradient behavior when relevant. Distinguish functions that build callables
from functions that evaluate values. Explain buffer donation and asynchronous
device execution at the boundaries where callers need to know about them.
Use executable examples only when their results can be checked. Comments
should explain mathematical choices or invariants rather than restate Python.

Measure documentation coverage, long lines and Radon complexity locally.
Coverage is not a measure of explanation quality; maintainability index is
also sensitive to comments and layout. Never pad comments or change numerical
formulas merely to improve a score. For documentation/formatting-only changes,
check parsed-code equivalence after removing docstrings, syntax compatibility,
public signatures and package contents. See the validation policy for the recorded baseline and validation approach.


## Development validation

Work targets the `development` branch. See the [validation policy](.github/validation/README.md)
for API snapshots, numerical references, measured coverage, dependency boundaries,
type/complexity/duplication baselines and performance reports. Changes must preserve
benchmark call signatures and shared OHAGI metadata inference.
