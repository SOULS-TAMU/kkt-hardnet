# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/SOULS-TAMU/kkt-hardnet/security/advisories/new).
Include affected versions, a minimal reproduction, impact and any suggested fix.
Do not disclose vulnerabilities or credentials in public issues.
Maintainers will assess the report and coordinate any fix and disclosure through
the private advisory. No response deadline or support guarantee is promised.

## Supported versions

Security fixes target the latest published stable release. Older releases are
not promised backported fixes. Upgrade when a security release becomes available.
Dependency and CodeQL checks help detect issues but do not guarantee their absence.

## Model files

Load model files and checkpoints only from sources you trust. Treat supplied
Python problem definitions and custom callables as executable code.
