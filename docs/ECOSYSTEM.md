# The HardNet ecosystem

The libraries have separate responsibilities and compatible public interfaces.

| Library | Responsibility | Current series |
|---|---|---|
| [KKT-HardNet](https://github.com/SOULS-TAMU/kkt-hardnet) | Surrogate, optimization and inverse-estimation problem supplier | 2.0 |
| [DAE-HardNet](https://github.com/SOULS-TAMU/dae-hardnet) | Differential-term processing and compiled problem supplier | 2.0 |
| [OHAGI](https://github.com/SOULS-TAMU/Ohagi) | Shared backbone, training, loss, checkpoint and inference execution | 1.0 |
| [PURIN](https://github.com/SOULS-TAMU/Purin) | Projection preparation, solvers, sparse operators and differentiation | 1.0 |

PURIN is usable independently. OHAGI depends on PURIN; the frontends use both.
Framework-specific problem definitions stay in the frontend. The shared engine
does not contain a different learning workflow for each benchmark.

NLPOptNet is another frontend in the ecosystem. Its nonlinear adapter constructs
local quadratic/linearized subproblems before using the shared numerical services.
This documentation release covers the four libraries above.
