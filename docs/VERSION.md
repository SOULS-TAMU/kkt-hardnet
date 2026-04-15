# Version Notes

## 0.2.0 - 2026-04-14

- Refactored the public builder API from `ProblemBuilder` to `KKTHardNet`
- Added dataset-driven entrypoints:
  - `model()`
  - `optimize()`
  - `estimate()`
- Added `metadata.json` save/load workflow
- Added `predict()` for trained and loaded models
- Removed the `case/` workflow and bundled factory runners from the public repo structure
- Switched the PyPI package name to `kkt-hardnet`
- Added Python 3.13-compatible dependency pins through environment markers
