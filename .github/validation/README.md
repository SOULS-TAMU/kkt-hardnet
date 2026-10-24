# Additional validation workflows

`validation.yml` runs on pull requests, pushes to main, manual dispatch and a
manual development runs. `dependency-review.yml` separately checks newly introduced
dependencies for moderate-or-higher known vulnerabilities. These workflows use
read-only repository permissions and do not post pull-request comments.

| Check | Policy |
| --- | --- |
| Public API | Compare existing public functions, methods, fields and lazy exports against `api.json`; additions are allowed. |
| Numerical references | Check deterministic short-run predictions against the saved frontend reference (atol=rtol=2e-5), analytic constraints and metadata round trips. PURIN owns solver-level numerical tests. |
| Frontend integration | Train the installed frontend briefly; verify physical constraints, full-model/facade loading and fresh-process OHAGI inference exports without frontend dependencies. |
| Cognitive complexity | Complexipy 8.1.0; new functions must score at most 15; existing functions cannot exceed their recorded score or 15. |
| Duplication | jscpd 5.4.2, exact clones of at least 80 tokens and six lines; no new clone fingerprints or increased duplicated-line count. |
| Type checking | mypy 2.4.0, Python 3.12/Linux target, imports skipped; no new error fingerprints or increased counts. This is a regression baseline, not a clean type-check claim. |
| Dependency boundaries | The frontend may import OHAGI/PURIN but cannot import another frontend; no new static import cycles. Static cycles include deferred and type-only imports. |
| Documentation | Validate local files/headings, parse BibTeX/CFF, build downloadable HTML, and check external content links with retries. Dynamic status badges are excluded; HTTP 429 is treated as rate limiting. |
| Performance | Compare base and candidate source in separate processes on the same runner/dependencies. Record compilation, synchronized singleton latency, batch throughput and memory; timing ratios are informational. |

The API and quality snapshots record the current state, including existing type
errors, complex functions and static cycles. Improvements are welcome. Baseline
updates require explicit maintainer review and a compatibility explanation; CI
never updates them automatically. They are not substitutes for numerical tests.

Performance workloads are small synthetic CPU cases, not application/MPC
benchmarks. The report distinguishes device-array and host-array inference,
keeps loading separate, and records Python allocation and process peak memory.
Shared-runner timings do not justify a strict latency regression threshold.
Artifacts retain both base/candidate observations for 90 days. Pull requests use
the base SHA; pushes use the previous SHA; scheduled/manual runs compare the same
source to show runner noise.

To reproduce checks, install `.github/validation/requirements.txt`, run
`python .github/scripts/api_contract.py`, then generate the jscpd JSON report
with the command in the workflow and run
`python .github/scripts/quality_metrics.py quality-duplication/jscpd-report.json`.
Run `documentation.py`, `numerical_regression.py` and `performance.py` from the
repository root with the package/dependencies installed. The frontend script
lives in OHAGI and validates fixture hashes before running.


This is the first shared-engine development snapshot, not an API equivalence claim
with the historical main branch. Existing numerical APIs are frozen from the local
shared-engine package. Performance reports benchmark the actual trained frontend
and distinguish loading, compilation, device calls and synchronized host calls.
The initial historical main implementation is not a comparable shared-engine
baseline; its performance is recorded as unavailable until a compatible base exists.
Coverage is measured inclusively and uses a reviewed initial floor; the long-term
coverage target is at least 85%. No production paths are excluded to inflate it.

GitHub scheduled workflows run only from the default branch. Development checks
run on pushes, pull requests targeting development and manual dispatch; the
CodeQL schedule becomes active if this workflow later reaches the default branch.

Initial inclusive coverage: 58.74%; enforced floor: 58%. Maximum cognitive
complexity: 54; six existing mypy errors; 0.19% exact duplicate lines; one
existing static import cycle; zero imports of other frontends.
