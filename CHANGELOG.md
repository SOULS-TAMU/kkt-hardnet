# Changelog

## Unreleased

- Use the renamed `ohagi` and `purin` dependencies without changing frontend APIs.

- Organize shared projection code in PURIN and shared learning services in OHAGI.
- Preserve frontend APIs and JAX metadata inference.
- Add distribution metadata and repository documentation.

## Shared engine and frontend ownership

Framework problem APIs and task policies live in their frontend packages. OHAGI executes all training through shared epoch plans and gradient/update builders; PURIN owns projection mathematics. Historical imports, benchmark call signatures, metadata formats, and checkpoint transport are preserved.

## Shared model bundles and frontend layouts

- DAEHN and NLPOPTNet now use canonical `api/` and `serialization/` directories,
  with historical problem/training imports retained through exact aliases.
- All new model saves use the shared OHAGI JAX bundle and numeric parameter
  codec. Redundant legacy writers, native/portable save switches, and standalone
  StableHLO/derivative exporters were removed.
- JAX serialization is private to OHAGI. Metadata-driven single-instance
  inference and separate Orbax checkpoints remain supported.

- Remove legacy saved-model readers; accept only shared OHAGI JAX bundles.

- Use OHAGI directly for projected-model conveniences and low-level serialization, retaining public signatures.


## Development validation

Work targets the `development` branch. See the [validation policy](.github/validation/README.md)
for API snapshots, numerical references, measured coverage, dependency boundaries,
type/complexity/duplication baselines and performance reports. Changes must preserve
benchmark call signatures and shared OHAGI metadata inference.

## Unreleased — coverage and quality
+
+- Raise inclusive statement-and-branch coverage enforcement to 90%.
+- Add numerical, gradient, serialization, differential, data and reporting tests.
+- Simplify host-side phase/reporting policies and differential stencil preparation
+  into focused private helpers; retain public signatures and numerical operations.
+- Resolve the recorded type-check errors and enforce cognitive complexity <=15.
+
