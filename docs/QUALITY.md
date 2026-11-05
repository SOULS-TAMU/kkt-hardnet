# Quality and compatibility

## Measured source structure

Local baseline, 2026-11-05. Scope: `src/` only; generated output, tests and
repository automation are excluded from these structural counts. The ecosystem
update changes no source files, so these values must remain identical afterward.

| Metric | Baseline | Policy |
| --- | ---: | --- |
| Dependency boundary violations | 0 | Zero new or existing violations |
| Static import cycles | 1 | No new cycles |
| Duplicate lines | 8 | No new duplicate fingerprints |
| Duplicate lines (%) | 0.18% | No increase |
| Maximum cognitive complexity | 15 | New functions ≤15; existing scores cannot increase |
| Mypy diagnostics | 0 | No new diagnostics |
| Functions measured | 126 | Informational |

Measurements use Complexipy 8.1.0, Mypy 2.4.0 and jscpd 5.4.2
(minimum 80 tokens and 6 lines). Dependency rules and static cycles are checked
by the repository's AST-based validation script. Counts describe the configured
checks, not a universal score for readability or correctness.

Existing complexity, typing or cycle debt is recorded in the version-controlled
validation baselines. A green check means no regression against those limits;
it does not mean every function is below 15 or every diagnostic is resolved.
Changing numerical code to address that debt requires a separate, reviewed change.

## Behavioral protection

- Public API snapshots and numerical reference checks retain their existing limits.
- The entire package remains in combined statement-and-branch coverage, with a
  minimum of **90%**. No modules or tests are excluded to raise the score.
- Existing OS/Python matrices and portable `metadata.json` inference checks remain.
- Timing measurements are informational and are not evidence of numerical equality.

## Reproduce and inspect

Run the commands in [the validation policy](../.github/validation/README.md).
The Validation workflow uploads `quality-results.json` and
`quality-duplication/jscpd-report.json` as `code-quality-metrics`.
The CI workflow publishes measured coverage separately. Workflow badges show
check status; they are not third-party certifications.

Shared composite actions are hosted in OHAGI and pinned to immutable commits.
They reuse the caller's own validation scripts and baselines. Publishing remains
in each repository's original workflow and environment, with upload permissions
limited to its publishing job.
