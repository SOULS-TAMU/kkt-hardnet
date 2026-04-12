from __future__ import annotations

import cvxpy as cp
import numpy as np

from scripts.misc.inequality_multipliers import default_ineq_multipliers


def _min_affine_over_box(row, x_L, x_U):
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    x_L = np.asarray(x_L, dtype=np.float64).reshape(-1)
    x_U = np.asarray(x_U, dtype=np.float64).reshape(-1)
    return float(np.sum(np.where(row >= 0.0, row * x_L, row * x_U)))


def _max_affine_over_box(row, x_L, x_U):
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    x_L = np.asarray(x_L, dtype=np.float64).reshape(-1)
    x_U = np.asarray(x_U, dtype=np.float64).reshape(-1)
    return float(np.sum(np.where(row >= 0.0, row * x_U, row * x_L)))


def _parameter_probe_points(
    x_L,
    x_U,
    *,
    include_vertices: bool = True,
    max_vertices: int = 4096,
    max_random: int = 8,
    seed: int = 0,
):
    x_L = np.asarray(x_L, dtype=np.float64).reshape(-1)
    x_U = np.asarray(x_U, dtype=np.float64).reshape(-1)
    p = int(x_L.shape[0])
    midpoint = 0.5 * (x_L + x_U)
    points = [midpoint, x_L, x_U]
    for j in range(p):
        probe_lo = midpoint.copy()
        probe_hi = midpoint.copy()
        probe_lo[j] = x_L[j]
        probe_hi[j] = x_U[j]
        points.append(probe_lo)
        points.append(probe_hi)

    if include_vertices and p > 0 and (1 << p) <= int(max_vertices):
        for mask in range(1 << p):
            vertex = x_L.copy()
            for bit in range(p):
                if (mask >> bit) & 1:
                    vertex[bit] = x_U[bit]
            points.append(vertex)

    rng = np.random.default_rng(int(seed))
    for _ in range(min(int(max_random), max(4, 2 * p))):
        points.append(rng.uniform(x_L, x_U))

    return np.asarray(points, dtype=np.float64)


def _dataset_preview_points(*, x_L, x_U, num_samples: int, seed: int) -> np.ndarray:
    if int(num_samples) <= 0:
        return np.zeros((0, int(np.asarray(x_L, dtype=np.float64).size)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    x_L = np.asarray(x_L, dtype=np.float64).reshape(-1)
    x_U = np.asarray(x_U, dtype=np.float64).reshape(-1)
    return np.asarray(rng.uniform(x_L, x_U, size=(int(num_samples), int(x_L.size))), dtype=np.float64)


def _make_dense_equality_block(rng: np.random.Generator, n_eq: int, n_y: int) -> np.ndarray:
    if int(n_eq) == 0:
        return np.zeros((0, int(n_y)), dtype=np.float64)

    q, _ = np.linalg.qr(rng.normal(size=(int(n_y), int(n_eq))))
    A = q.T
    row_scales = 0.8 + 0.4 * rng.uniform(size=(int(n_eq),))
    row_signs = np.where(rng.uniform(size=(int(n_eq),)) < 0.5, -1.0, 1.0)
    return (row_scales * row_signs)[:, None] * A


def _make_feasible_affine_map(
    rng: np.random.Generator,
    *,
    n_y: int,
    n_x: int,
    x_L,
    x_U,
) -> tuple[np.ndarray, np.ndarray]:
    x_abs_max = np.maximum(np.abs(np.asarray(x_L, dtype=np.float64)), np.abs(np.asarray(x_U, dtype=np.float64)))
    feasible_center = -(1.25 + 0.25 * np.abs(rng.normal(size=(int(n_y),))))
    feasible_map = 0.12 * rng.normal(size=(int(n_y), int(n_x))) / max(1.0, np.sqrt(float(max(1, int(n_x)))))
    if feasible_map.size > 0:
        worst_case = np.abs(feasible_map) @ x_abs_max
        max_worst_case = float(np.max(worst_case)) if worst_case.size else 0.0
        if max_worst_case > 0.6:
            feasible_map *= 0.6 / max_worst_case
    return np.asarray(feasible_center, dtype=np.float64), np.asarray(feasible_map, dtype=np.float64)


def _make_equality_optimizer_map(Q, c, A, b, B) -> tuple[np.ndarray, np.ndarray]:
    Q = np.asarray(Q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    B = np.asarray(B, dtype=np.float64)
    n_y = int(c.size)
    n_x = int(B.shape[1]) if B.ndim == 2 else 0
    if A.shape[0] == 0:
        return -np.linalg.solve(Q, c), np.zeros((n_y, n_x), dtype=np.float64)

    q_inv_c = np.linalg.solve(Q, c)
    q_inv_a_t = np.linalg.solve(Q, A.T)
    schur = A @ q_inv_a_t
    lam_const = np.linalg.solve(schur, b + A @ q_inv_c)
    lam_map = np.linalg.solve(schur, B)
    y_const = -q_inv_c + q_inv_a_t @ lam_const
    y_map = q_inv_a_t @ lam_map
    return np.asarray(y_const, dtype=np.float64), np.asarray(y_map, dtype=np.float64)


def _cluster_preview_points(
    preview_X: np.ndarray,
    *,
    num_clusters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    preview_X = np.asarray(preview_X, dtype=np.float64)
    if preview_X.ndim != 2:
        raise ValueError(f"Expected a 2D preview matrix, received shape {preview_X.shape}.")

    num_points, n_x = preview_X.shape
    if num_points == 0:
        return np.zeros((0, n_x), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    k = max(1, min(int(num_clusters), num_points))
    centers = np.zeros((k, n_x), dtype=np.float64)
    first_idx = int(np.argmax(np.linalg.norm(preview_X, axis=1)))
    centers[0] = preview_X[first_idx]
    closest_dist_sq = np.sum((preview_X - centers[0]) ** 2, axis=1)

    for idx in range(1, k):
        next_idx = int(np.argmax(closest_dist_sq))
        centers[idx] = preview_X[next_idx]
        candidate_dist_sq = np.sum((preview_X - centers[idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, candidate_dist_sq)

    assignments = np.zeros((num_points,), dtype=np.int64)
    for _ in range(8):
        dist_sq = np.sum((preview_X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dist_sq, axis=1).astype(np.int64, copy=False)
        for idx in range(k):
            mask = assignments == idx
            if np.any(mask):
                centers[idx] = np.mean(preview_X[mask], axis=0)
            else:
                centers[idx] = preview_X[int(rng.integers(num_points))]

    return centers, assignments


def _make_affine_bounds(
    rng: np.random.Generator,
    *,
    feasible_center: np.ndarray,
    feasible_map: np.ndarray,
    x_L,
    x_U,
    bound_margin: float,
    bound_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feasible_center = np.asarray(feasible_center, dtype=np.float64).reshape(-1)
    feasible_map = np.asarray(feasible_map, dtype=np.float64)
    n_y, n_x = feasible_map.shape
    x_abs_max = np.maximum(np.abs(np.asarray(x_L, dtype=np.float64)), np.abs(np.asarray(x_U, dtype=np.float64)))

    coeff_scale = 0.05 * float(bound_scale) / max(1.0, np.sqrt(float(max(1, n_x))))
    lower_M = coeff_scale * rng.normal(size=(n_y, n_x))
    upper_M = coeff_scale * rng.normal(size=(n_y, n_x))
    if n_x > 0:
        max_budget = max(0.2 * max(float(bound_margin), 1.0), 1e-6)
        for mat in (lower_M, upper_M):
            worst_case = np.abs(mat) @ x_abs_max
            max_worst = float(np.max(worst_case)) if worst_case.size else 0.0
            if max_worst > max_budget:
                mat *= max_budget / max_worst

    l = np.zeros((n_y,), dtype=np.float64)
    u = np.zeros((n_y,), dtype=np.float64)
    neg_eps = 1e-3
    target_upper_max = -max(0.1, 0.1 * float(bound_margin))
    upper_gap = max(0.15, 0.25 * float(bound_margin))
    lower_gap = max(0.3, 0.4 * float(bound_margin))
    for j in range(n_y):
        max_upper_aff = _max_affine_over_box(upper_M[j], x_L, x_U)
        min_upper_aff = _min_affine_over_box(upper_M[j], x_L, x_U)
        upper_span = max_upper_aff - min_upper_aff
        required_upper_target = float(feasible_center[j] + upper_gap + upper_span)
        upper_target = min(-neg_eps, max(target_upper_max, required_upper_target))
        u[j] = upper_target - max_upper_aff
        l[j] = feasible_center[j] - _max_affine_over_box(lower_M[j], x_L, x_U) - lower_gap

    return lower_M, l, upper_M, u


def _evaluate_nonlinear_terms(Y, *, a_vec: np.ndarray, W_mat: np.ndarray):
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    exp_term = np.exp(Y) @ np.asarray(a_vec, dtype=np.float64).reshape(-1)
    quad_term = np.einsum("bi,ij,bj->b", Y, np.asarray(W_mat, dtype=np.float64), Y)
    return exp_term, quad_term, exp_term + quad_term


def _build_nlp_candidate(
    *,
    a_vec: np.ndarray,
    W_mat: np.ndarray,
    preview_X: np.ndarray,
    preview_eq_y: np.ndarray,
    support_X: np.ndarray,
    support_feas_y: np.ndarray,
    x_coeff: np.ndarray,
    rhs_margin: float,
):
    x_coeff = np.asarray(x_coeff, dtype=np.float64).reshape(-1)
    _, _, support_lhs = _evaluate_nonlinear_terms(support_feas_y, a_vec=a_vec, W_mat=W_mat)
    rhs = float(np.max(support_lhs - support_X @ x_coeff) + float(rhs_margin))

    preview_exp, preview_quad, preview_lhs = _evaluate_nonlinear_terms(preview_eq_y, a_vec=a_vec, W_mat=W_mat)
    preview_residuals = preview_lhs - (rhs + preview_X @ x_coeff)
    return {
        "a": np.asarray(a_vec, dtype=np.float64),
        "W": np.asarray(W_mat, dtype=np.float64),
        "rhs": rhs,
        "x_coeff": x_coeff,
        "preview_residuals": np.asarray(preview_residuals, dtype=np.float64),
        "preview_exp": np.asarray(preview_exp, dtype=np.float64),
        "preview_quad": np.asarray(preview_quad, dtype=np.float64),
    }


class NLPGenerator:
    """
    Parametric NLP generator of the form

        minimize_y    0.5 y^T Q y + c^T y
        subject to    A y = b + B x
                      a_i^T exp(y) + y^T W_i y <= beta_i + E_i x
                      l + L x <= y <= u + U x
    """

    def __init__(
        self,
        n_y: int,
        n_x: int,
        n_eq: int = 2,
        n_ineq: int = 3,
        seed: int = 0,
        is_diag_Q: bool = False,
    ):
        self.n_y = int(n_y)
        self.n_x = int(n_x)
        self.n_eq = int(n_eq)
        self.n_ineq = int(n_ineq)
        self.is_diag_Q = bool(is_diag_Q)
        self.seed = int(seed)
        self.parameter_seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.Q = None
        self.c = None
        self.A = None
        self.b = None
        self.B = None
        self.a = None
        self.W = None
        self.beta = None
        self.E = None
        self.l = None
        self.L = None
        self.u = None
        self.U = None
        self.x_L = None
        self.x_U = None
        self.y_feas = None

        self.problem = None
        self.y_var = None
        self.x_param = None
        self.ineq_constraints = None
        self.solver = "SCS"
        self.requested_solver = "cvxpy"
        self._pyo = None
        self.nonconvex_model = None
        self.nonconvex_solver = None

    def _make_objective_matrix(self, q_diag_shift=0.5):
        if self.is_diag_Q:
            return np.diag(0.8 + self.rng.random(self.n_y))
        M = self.rng.standard_normal((self.n_y, self.n_y))
        return (M.T @ M) / max(1, self.n_y) + (0.8 + float(q_diag_shift)) * np.eye(self.n_y)

    def _ensure_pyomo(self):
        if self._pyo is None:
            import pyomo.environ as pyo

            self._pyo = pyo
        return self._pyo

    def _ensure_gurobi(self):
        import gurobipy as gp

        return gp

    def set_solver(self, solver_name: str):
        solver_name = str(solver_name).strip().upper()
        if not solver_name:
            raise ValueError("Solver name cannot be empty.")
        self.solver = solver_name
        if solver_name == "GUROBI":
            self.requested_solver = "gurobi"
        elif solver_name == "IPOPT":
            self.requested_solver = "ipopt"
        else:
            self.requested_solver = "cvxpy"

    def set_solver_backend(self, solver_name: str):
        solver_name = str(solver_name).strip().lower()
        if solver_name == "gurobi":
            self.set_solver("GUROBI")
            return
        if solver_name == "ipopt":
            self.set_solver("IPOPT")
            return
        if solver_name in {"auto", "cvxpy"}:
            self.requested_solver = "cvxpy"
            return
        raise ValueError(f"Unsupported solver backend: {solver_name}")

    def _ensure_nonconvex_solver(self, solver_opts=None):
        pyo = self._ensure_pyomo()
        if self.nonconvex_solver is None:
            solver = pyo.SolverFactory("ipopt")
            if solver is None or not solver.available(exception_flag=False):
                raise RuntimeError("Ipopt solver is not available. Ensure `ipopt` is installed and on PATH.")
            self.nonconvex_solver = solver
        defaults = {"tol": 1e-8, "max_iter": 1000, "print_level": 0}
        options = dict(defaults)
        if solver_opts:
            options.update(solver_opts)
        try:
            self.nonconvex_solver.options.clear()
        except Exception:
            pass
        for key, value in options.items():
            self.nonconvex_solver.options[str(key)] = value
        return pyo, self.nonconvex_solver

    def _build_nonconvex_model(self):
        pyo = self._ensure_pyomo()
        m = pyo.ConcreteModel()
        m.J = pyo.RangeSet(0, self.n_y - 1)
        m.y = pyo.Var(m.J, initialize=0.0)

        if self.n_eq > 0:
            m.EQ = pyo.RangeSet(0, self.n_eq - 1)
            m.eq_rhs = pyo.Param(m.EQ, mutable=True, initialize=0.0)

            def eq_rule(model, i):
                return sum(self.A[i, j] * model.y[j] for j in range(self.n_y)) == model.eq_rhs[i]

            m.eq_con = pyo.Constraint(m.EQ, rule=eq_rule)

        if self.n_ineq > 0:
            m.IN = pyo.RangeSet(0, self.n_ineq - 1)
            m.nl_rhs = pyo.Param(m.IN, mutable=True, initialize=0.0)

            def nl_rule(model, i):
                exp_term = sum(self.a[i, j] * pyo.exp(model.y[j]) for j in range(self.n_y))
                quad_term = sum(self.W[i, r, c] * model.y[r] * model.y[c] for r in range(self.n_y) for c in range(self.n_y))
                return exp_term + quad_term <= model.nl_rhs[i]

            m.nl_con = pyo.Constraint(m.IN, rule=nl_rule)

        Qsym = 0.5 * (self.Q + self.Q.T)

        def obj_rule(model):
            quad = 0.5 * sum(Qsym[i, j] * model.y[i] * model.y[j] for i in range(self.n_y) for j in range(self.n_y))
            lin = sum(self.c[i] * model.y[i] for i in range(self.n_y))
            return quad + lin

        m.obj = pyo.Objective(rule=obj_rule)
        self.nonconvex_model = m

    def _initial_points(self, lb, ub):
        starts = [np.clip(np.zeros(self.n_y), lb, ub), np.clip(0.5 * (lb + ub), lb, ub)]
        if self.y_feas is not None:
            starts.append(np.clip(self.y_feas, lb, ub))
        for _ in range(4):
            starts.append(self.rng.uniform(lb, ub))
        return starts

    def _recover_kkt_ineq_multipliers(self, *, x_value, y_value, rhs_nl, lb, ub) -> np.ndarray:
        mu = np.zeros((self.n_ineq,), dtype=np.float64)
        if self.n_ineq <= 0:
            return mu

        try:
            from scipy.optimize import lsq_linear
        except Exception:
            return default_ineq_multipliers(self.n_ineq)

        y_value = np.asarray(y_value, dtype=np.float64).reshape(-1)
        x_value = np.asarray(x_value, dtype=np.float64).reshape(-1)
        rhs_nl = np.asarray(rhs_nl, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        exp_y = np.exp(y_value)
        grad_f = 0.5 * (self.Q + self.Q.T) @ y_value + self.c

        nl_resid = self.a @ exp_y + np.einsum("i,mij,j->m", y_value, self.W, y_value) - rhs_nl
        active_tol = 1e-6
        active_nl = np.where(nl_resid >= -active_tol)[0]
        active_lb = np.where(y_value - lb <= active_tol)[0]
        active_ub = np.where(ub - y_value <= active_tol)[0]

        blocks = []
        lower = []
        upper = []

        if self.n_eq > 0:
            blocks.append(np.asarray(self.A, dtype=np.float64).T)
            lower.extend([-np.inf] * self.n_eq)
            upper.extend([np.inf] * self.n_eq)

        if active_nl.size > 0:
            w_sym = np.asarray(self.W[active_nl], dtype=np.float64)
            w_sym = w_sym + np.swapaxes(w_sym, 1, 2)
            grad_nl = np.asarray(self.a[active_nl], dtype=np.float64) * exp_y[None, :]
            grad_nl = grad_nl + np.einsum("mij,j->mi", w_sym, y_value)
            blocks.append(grad_nl.T)
            lower.extend([0.0] * int(active_nl.size))
            upper.extend([np.inf] * int(active_nl.size))

        if active_lb.size > 0:
            blocks.append(-np.eye(self.n_y, dtype=np.float64)[:, active_lb])
            lower.extend([0.0] * int(active_lb.size))
            upper.extend([np.inf] * int(active_lb.size))

        if active_ub.size > 0:
            blocks.append(np.eye(self.n_y, dtype=np.float64)[:, active_ub])
            lower.extend([0.0] * int(active_ub.size))
            upper.extend([np.inf] * int(active_ub.size))

        if not blocks:
            return mu

        matrix = np.concatenate(blocks, axis=1)
        try:
            result = lsq_linear(
                matrix,
                -grad_f,
                bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
                method="trf",
                lsmr_tol="auto",
                verbose=0,
            )
        except Exception:
            return default_ineq_multipliers(self.n_ineq)

        if result.x.size == 0:
            return mu

        offset = self.n_eq
        if active_nl.size > 0:
            mu_active = np.asarray(result.x[offset : offset + active_nl.size], dtype=np.float64)
            finite = np.isfinite(mu_active)
            mu_active[finite] = np.maximum(mu_active[finite], 0.0)
            mu[active_nl] = mu_active
        return mu

    def _preview_box_qp_solutions(self, preview_X: np.ndarray, fallback_Y: np.ndarray) -> np.ndarray:
        preview_X = np.asarray(preview_X, dtype=np.float64)
        fallback_Y = np.asarray(fallback_Y, dtype=np.float64)
        if preview_X.size == 0:
            return fallback_Y

        y = cp.Variable(self.n_y)
        x = cp.Parameter(self.n_x)
        constraints = []
        if self.n_eq > 0:
            constraints.append(self.A @ y == self.b + self.B @ x)
        constraints.extend([y >= self.l + self.L @ x, y <= self.u + self.U @ x])
        Qsym = 0.5 * (self.Q + self.Q.T)
        problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, Qsym) + self.c @ y), constraints)

        solutions = []
        for idx, x_value in enumerate(preview_X):
            x.value = x_value
            try:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception:
                pass
            if problem.status in {"optimal", "optimal_inaccurate"} and y.value is not None:
                solutions.append(np.asarray(y.value, dtype=np.float64).reshape(-1))
            else:
                solutions.append(np.asarray(fallback_Y[idx], dtype=np.float64).reshape(-1))
        return np.asarray(solutions, dtype=np.float64)

    def _preview_box_min_norm_solutions(self, preview_X: np.ndarray, fallback_Y: np.ndarray) -> np.ndarray:
        preview_X = np.asarray(preview_X, dtype=np.float64)
        fallback_Y = np.asarray(fallback_Y, dtype=np.float64)
        if preview_X.size == 0:
            return fallback_Y

        y = cp.Variable(self.n_y)
        x = cp.Parameter(self.n_x)
        constraints = []
        if self.n_eq > 0:
            constraints.append(self.A @ y == self.b + self.B @ x)
        constraints.extend([y >= self.l + self.L @ x, y <= self.u + self.U @ x])
        problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(y)), constraints)

        solutions = []
        for idx, x_value in enumerate(preview_X):
            x.value = x_value
            try:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception:
                pass
            if problem.status in {"optimal", "optimal_inaccurate"} and y.value is not None:
                solutions.append(np.asarray(y.value, dtype=np.float64).reshape(-1))
            else:
                solutions.append(np.asarray(fallback_Y[idx], dtype=np.float64).reshape(-1))
        return np.asarray(solutions, dtype=np.float64)

    def build_problem_data(
        self,
        x_L,
        x_U,
        q_diag_shift=0.5,
        nl_margin=1.0,
        bound_margin=1.0,
        bound_scale=0.2,
        param_scale=0.4,
        preview_num_samples: int | None = None,
    ):
        del param_scale

        x_L = np.asarray(x_L, dtype=float).reshape(-1)
        x_U = np.asarray(x_U, dtype=float).reshape(-1)
        if x_L.shape[0] != self.n_x or x_U.shape[0] != self.n_x:
            raise ValueError("x_L and x_U must have length n_x.")
        if np.any(x_L > x_U):
            raise ValueError("Each component of x_L must be <= x_U.")

        self.x_L = x_L
        self.x_U = x_U
        self.problem = None
        self.nonconvex_model = None

        feasible_center, feasible_map = _make_feasible_affine_map(
            self.rng,
            n_y=self.n_y,
            n_x=self.n_x,
            x_L=self.x_L,
            x_U=self.x_U,
        )
        self.y_feas = feasible_center.copy()
        self.Q = self._make_objective_matrix(q_diag_shift=q_diag_shift)
        self.c = 0.5 * self.rng.standard_normal(self.n_y)

        self.A = _make_dense_equality_block(self.rng, self.n_eq, self.n_y)
        if self.n_eq > 0:
            self.B = np.zeros((self.n_eq, self.n_x), dtype=np.float64)
            self.b = self.A @ feasible_center
        else:
            self.B = np.zeros((0, self.n_x), dtype=np.float64)
            self.b = np.zeros((0,), dtype=np.float64)

        self.L, self.l, self.U, self.u = _make_affine_bounds(
            self.rng,
            feasible_center=feasible_center,
            feasible_map=feasible_map,
            x_L=self.x_L,
            x_U=self.x_U,
            bound_margin=float(bound_margin),
            bound_scale=float(bound_scale),
        )

        eq_opt_center, eq_opt_map = _make_equality_optimizer_map(self.Q, self.c, self.A, self.b, self.B)
        preview_count = int(preview_num_samples) if preview_num_samples is not None else 0
        preview_X = _dataset_preview_points(
            x_L=self.x_L,
            x_U=self.x_U,
            num_samples=preview_count,
            seed=self.parameter_seed,
        )
        if preview_X.shape[0] == 0:
            preview_X = _parameter_probe_points(
                self.x_L,
                self.x_U,
                include_vertices=False,
                max_random=max(8, 2 * self.n_x),
                seed=self.parameter_seed,
            )

        preview_eq_y = eq_opt_center[None, :] + preview_X @ eq_opt_map.T
        preview_eq_y = self._preview_box_qp_solutions(preview_X, preview_eq_y)
        preview_feas_y = self._preview_box_min_norm_solutions(preview_X, preview_eq_y)
        support_X = _parameter_probe_points(
            self.x_L,
            self.x_U,
            include_vertices=True,
            max_random=0,
            seed=self.parameter_seed + 17,
        )
        support_feas_y = (
            self._preview_box_min_norm_solutions(
                support_X,
                np.repeat(feasible_center[None, :], support_X.shape[0], axis=0),
            )
            if support_X.size
            else np.zeros((0, self.n_y), dtype=np.float64)
        )
        activity_tol = 1e-3

        if preview_X.shape[0] > 0:
            centers, assignments = _cluster_preview_points(
                preview_X,
                num_clusters=min(self.n_ineq, preview_X.shape[0]),
                rng=self.rng,
            )
        else:
            centers = np.zeros((0, self.n_x), dtype=np.float64)
            assignments = np.zeros((0,), dtype=np.int64)
        global_x_mean = np.mean(preview_X, axis=0) if preview_X.shape[0] > 0 else np.zeros((self.n_x,), dtype=np.float64)

        def _support_weights(eq_group, feas_group):
            eq_ref = np.median(eq_group, axis=0)
            feas_ref = np.median(feas_group, axis=0)
            gain = np.maximum(np.mean(eq_group ** 2 - feas_group ** 2, axis=0), 0.0)
            if float(np.max(gain)) <= 1e-6:
                gain = np.maximum(np.abs(eq_ref) - np.abs(feas_ref), 0.0)
            if float(np.max(gain)) <= 1e-6:
                gain = np.abs(eq_ref - feas_ref)
            if float(np.max(gain)) <= 1e-6:
                gain = np.maximum(np.mean(np.abs(eq_group), axis=0), 1e-3)

            support_size = max(2, min(4, self.n_y))
            support = np.argpartition(-gain, support_size - 1)[:support_size]
            support = support[np.argsort(-gain[support])]
            eq_sq = np.maximum(np.median(eq_group[:, support] ** 2, axis=0), 1e-4)
            raw_w = np.maximum(gain[support], 1e-3)
            return support, eq_ref, raw_w, eq_sq

        def _build_primary_candidate():
            if preview_X.shape[0] == 0:
                return None

            support, eq_ref, raw_w, eq_sq = _support_weights(preview_eq_y, preview_feas_y)
            target_quad = 12.0
            base_scale = target_quad / max(float(np.dot(raw_w, eq_sq)), 1e-6)
            best = None
            best_floor = -np.inf
            for boost in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
                w_diag = np.zeros((self.n_y,), dtype=np.float64)
                w_diag[support] = boost * base_scale * raw_w
                W_mat = np.diag(w_diag)
                quad_ref = float(eq_ref @ W_mat @ eq_ref)
                desired_exp = float(np.clip(0.00015 * max(quad_ref, target_quad), 0.0005, 0.002))
                a_vec = np.zeros((self.n_y,), dtype=np.float64)
                raw_exp = np.exp(-np.clip(eq_ref[support], -1.5, 1.5))
                denom = float(np.sum(raw_exp * np.exp(eq_ref[support])))
                if denom <= 1e-8:
                    raw_exp = np.ones_like(raw_exp)
                    denom = float(np.sum(np.exp(eq_ref[support])))
                a_vec[support] = desired_exp * raw_exp / max(denom, 1e-8)
                candidate = _build_nlp_candidate(
                    a_vec=a_vec,
                    W_mat=W_mat,
                    preview_X=preview_X,
                    preview_eq_y=preview_eq_y,
                    support_X=support_X,
                    support_feas_y=support_feas_y,
                    x_coeff=np.zeros((self.n_x,), dtype=np.float64),
                    rhs_margin=float(nl_margin),
                )
                floor = float(np.min(candidate["preview_residuals"])) if candidate["preview_residuals"].size > 0 else -np.inf
                if floor > best_floor:
                    best_floor = floor
                    best = candidate
                if floor > 5e-2:
                    return candidate
            return best

        def _best_candidate(mask, center_x):
            eq_group = preview_eq_y[mask]
            feas_group = preview_feas_y[mask]
            if eq_group.size == 0 or feas_group.size == 0:
                return None

            support, eq_ref, raw_w, eq_sq = _support_weights(eq_group, feas_group)
            desired_quad = float(np.clip(12.0 + 0.75 * np.median(np.sum(eq_group[:, support] ** 2, axis=1)), 10.0, 16.0))
            scale = desired_quad / max(float(np.dot(raw_w, eq_sq)), 1e-6)
            w_diag = np.zeros((self.n_y,), dtype=np.float64)
            w_diag[support] = scale * raw_w
            W_mat = np.diag(w_diag)

            quad_ref = float(eq_ref @ W_mat @ eq_ref)
            desired_exp = float(np.clip(0.002 * max(quad_ref, desired_quad), 0.005, 0.025))
            a_vec = np.zeros((self.n_y,), dtype=np.float64)
            raw_exp = np.exp(-np.clip(eq_ref[support], -1.5, 1.5))
            denom = float(np.sum(raw_exp * np.exp(eq_ref[support])))
            if denom <= 1e-8:
                raw_exp = np.ones_like(raw_exp)
                denom = float(np.sum(np.exp(eq_ref[support])))
            a_vec[support] = desired_exp * raw_exp / max(denom, 1e-8)

            raw_param_dir = np.asarray(center_x - global_x_mean, dtype=np.float64)
            param_norm = float(np.linalg.norm(raw_param_dir))
            candidate_dirs = [np.zeros((self.n_x,), dtype=np.float64)]
            if param_norm > 1e-12:
                unit_dir = raw_param_dir / param_norm
                for scale_dir in (0.25, 0.5, 0.75, 1.0, 1.5):
                    candidate_dirs.append(scale_dir * unit_dir)
                    candidate_dirs.append(-scale_dir * unit_dir)

            best = None
            best_score = (-1, -np.inf, -np.inf, -np.inf)
            for x_coeff in candidate_dirs:
                candidate = _build_nlp_candidate(
                    a_vec=a_vec,
                    W_mat=W_mat,
                    preview_X=preview_X,
                    preview_eq_y=preview_eq_y,
                    support_X=support_X,
                    support_feas_y=support_feas_y,
                    x_coeff=x_coeff,
                    rhs_margin=float(nl_margin),
                )
                inside = candidate["preview_residuals"][mask]
                outside = candidate["preview_residuals"][~mask]
                inside_active = int(np.sum(inside > activity_tol))
                outside_active = int(np.sum(outside > activity_tol))
                inside_rate = float(inside_active) / max(int(inside.size), 1)
                outside_rate = float(outside_active) / max(int(outside.size), 1)
                inside_median = float(np.median(inside)) if inside.size > 0 else -np.inf
                outside_high = float(np.quantile(outside, 0.75)) if outside.size > 0 else -np.inf
                exp_med = float(np.median(candidate["preview_exp"][mask])) if inside.size > 0 else np.inf
                quad_med = float(np.median(candidate["preview_quad"][mask])) if inside.size > 0 else 0.0
                ratio_penalty = -float(exp_med / max(quad_med, 1e-6))
                score = (
                    inside_rate - outside_rate,
                    inside_active - outside_active,
                    inside_median - outside_high,
                    ratio_penalty,
                )
                if score > best_score:
                    best_score = score
                    best = candidate

            if best is None:
                return None

            inside = best["preview_residuals"][mask]
            outside = best["preview_residuals"][~mask]
            if inside.size > 0 and outside.size > 0:
                inside_budget = float(max(np.quantile(inside, 0.4) - activity_tol, 0.0))
                outside_budget = float(max(np.quantile(outside, 0.75), 0.0))
                rhs_shift = min(inside_budget, outside_budget)
                if rhs_shift > 0.0:
                    best["rhs"] = float(best["rhs"] + rhs_shift)
                    best["preview_residuals"] = best["preview_residuals"] - rhs_shift

            return best

        selected = []
        for cluster_idx, center_x in enumerate(centers):
            mask = assignments == cluster_idx
            candidate = _best_candidate(mask, center_x)
            if candidate is not None:
                selected.append(candidate)

        if selected and preview_X.shape[0] > 0:
            cover_scores = np.max(np.stack([cand["preview_residuals"] for cand in selected], axis=0), axis=0)
            uncovered = np.where(cover_scores <= activity_tol)[0]
            for idx in uncovered.tolist():
                if len(selected) >= self.n_ineq:
                    break
                point_mask = np.zeros((preview_X.shape[0],), dtype=bool)
                point_mask[idx] = True
                candidate = _best_candidate(point_mask, preview_X[idx])
                if candidate is not None:
                    selected.append(candidate)

        if not selected or (len(selected) < self.n_ineq and preview_X.shape[0] > 0):
            primary_candidate = _build_primary_candidate()
            if primary_candidate is not None:
                selected.append(primary_candidate)

        if not selected:
            zero_a = np.zeros((self.n_y,), dtype=np.float64)
            zero_W = np.zeros((self.n_y, self.n_y), dtype=np.float64)
            selected = [
                _build_nlp_candidate(
                    a_vec=zero_a,
                    W_mat=zero_W,
                    preview_X=preview_X,
                    preview_eq_y=preview_eq_y,
                    support_X=support_X,
                    support_feas_y=support_feas_y,
                    x_coeff=np.zeros((self.n_x,), dtype=np.float64),
                    rhs_margin=float(nl_margin),
                )
            ]

        active_templates = list(selected[: self.n_ineq])
        while len(active_templates) < self.n_ineq:
            base = selected[(len(active_templates) - len(selected)) % len(selected)]
            relax_idx = len(active_templates) - len(selected)
            active_templates.append(
                {
                    "a": base["a"].copy(),
                    "W": base["W"].copy(),
                    "rhs": float(base["rhs"] + 1.0 + 0.25 * relax_idx),
                    "x_coeff": base["x_coeff"].copy(),
                    "preview_residuals": base["preview_residuals"].copy(),
                }
            )

        self.a = np.stack([cand["a"] for cand in active_templates[: self.n_ineq]], axis=0)
        self.W = np.stack([cand["W"] for cand in active_templates[: self.n_ineq]], axis=0)
        self.beta = np.asarray([cand["rhs"] for cand in active_templates[: self.n_ineq]], dtype=np.float64)
        self.E = np.stack([cand["x_coeff"] for cand in active_templates[: self.n_ineq]], axis=0)
        return self.get_problem_data()

    def build_cvxpy_problem(self, solver="SCS"):
        self.solver = str(solver).strip().upper()
        y = cp.Variable(self.n_y)
        x = cp.Parameter(self.n_x)
        constraints = []
        ineq_constraints = []

        if self.n_eq > 0:
            constraints.append(self.A @ y == self.b + self.B @ x)

        constraints.extend(
            [
                y >= self.l + self.L @ x,
                y <= self.u + self.U @ x,
            ]
        )

        for i in range(self.n_ineq):
            con = self.a[i] @ cp.exp(y) + cp.quad_form(y, 0.5 * (self.W[i] + self.W[i].T)) <= self.beta[i] + self.E[i] @ x
            constraints.append(con)
            ineq_constraints.append(con)

        Qsym = 0.5 * (self.Q + self.Q.T)
        objective = cp.Minimize(0.5 * cp.quad_form(y, Qsym) + self.c @ y)

        self.problem = cp.Problem(objective, constraints)
        self.y_var = y
        self.x_param = x
        self.ineq_constraints = ineq_constraints

    def solve_for_x(self, x_value, solver_opts=None):
        if self.requested_solver == "ipopt":
            return self._solve_nonconvex_for_x(x_value, solver_opts=solver_opts)
        if self.requested_solver == "gurobi":
            return self._solve_with_gurobi(x_value, solver_opts=solver_opts)

        if self.problem is None:
            raise RuntimeError("Call build_cvxpy_problem(...) first.")

        x_value = np.asarray(x_value, dtype=float).reshape(-1)
        if x_value.shape[0] != self.n_x:
            raise ValueError("x_value must have length n_x.")

        self.x_param.value = x_value
        kwargs = {}
        if solver_opts is not None:
            kwargs.update(solver_opts)

        try:
            self.problem.solve(solver=self.solver, **kwargs)
        except cp.SolverError:
            return {
                "status": "solver_error",
                "objective": None,
                "y": None,
                "mu": default_ineq_multipliers(self.n_ineq),
                "solve_time_sec": None,
                "solver": self.solver,
            }

        mu = default_ineq_multipliers(self.n_ineq)
        if self.n_ineq > 0 and self.ineq_constraints is not None:
            mu = np.asarray([constraint.dual_value for constraint in self.ineq_constraints], dtype=np.float64).reshape(-1)
            finite = np.isfinite(mu)
            mu[finite] = np.maximum(mu[finite], 0.0)

        return {
            "status": self.problem.status,
            "objective": None if self.problem.value is None else float(self.problem.value),
            "y": None if self.y_var.value is None else np.asarray(self.y_var.value).reshape(-1),
            "mu": mu,
            "solve_time_sec": float(self.problem.solver_stats.solve_time)
            if getattr(self.problem, "solver_stats", None) is not None
            and getattr(self.problem.solver_stats, "solve_time", None) is not None
            else None,
            "solver": self.solver,
        }

    def _solve_with_gurobi(self, x_value, solver_opts=None):
        x_value = np.asarray(x_value, dtype=float).reshape(-1)
        if x_value.shape[0] != self.n_x:
            raise ValueError("x_value must have length n_x.")

        rhs_eq = self.b + self.B @ x_value
        rhs_nl = self.beta + self.E @ x_value
        lb = self.l + self.L @ x_value
        ub = self.u + self.U @ x_value
        Qsym = 0.5 * (self.Q + self.Q.T)

        gp = self._ensure_gurobi()
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.QCPDual = 1
        if solver_opts:
            for key, value in solver_opts.items():
                m.setParam(str(key), value)

        y = m.addVars(self.n_y, lb=lb.tolist(), ub=ub.tolist(), name="y")
        z = m.addVars(self.n_y, lb=0.0, name="zexp")
        for j in range(self.n_y):
            m.addGenConstrExp(y[j], z[j], name=f"exp_{j}")

        if self.n_eq > 0:
            for i in range(self.n_eq):
                m.addConstr(sum(self.A[i, j] * y[j] for j in range(self.n_y)) == float(rhs_eq[i]))

        q_constraints = []
        if self.n_ineq > 0:
            for i in range(self.n_ineq):
                expr = sum(self.a[i, j] * z[j] for j in range(self.n_y))
                expr += sum(self.W[i, r, c] * y[r] * y[c] for r in range(self.n_y) for c in range(self.n_y))
                q_constraints.append(m.addQConstr(expr <= float(rhs_nl[i])))

        obj = sum(0.5 * Qsym[i, j] * y[i] * y[j] for i in range(self.n_y) for j in range(self.n_y))
        obj += sum(self.c[j] * y[j] for j in range(self.n_y))
        m.setObjective(obj, gp.GRB.MINIMIZE)
        m.optimize()

        if m.SolCount <= 0:
            return {
                "status": "solver_error",
                "objective": None,
                "y": None,
                "mu": default_ineq_multipliers(self.n_ineq),
                "solve_time_sec": float(getattr(m, "Runtime", 0.0)),
            }

        y_sol = np.asarray([y[j].X for j in range(self.n_y)], dtype=float)
        status = "optimal" if m.Status == gp.GRB.OPTIMAL else "optimal_inaccurate"
        mu = default_ineq_multipliers(self.n_ineq)
        if self.n_ineq > 0:
            try:
                mu = np.asarray([-float(qc.QCPi) for qc in q_constraints], dtype=np.float64)
                finite = np.isfinite(mu)
                mu[finite] = np.maximum(mu[finite], 0.0)
            except Exception:
                mu = self._recover_kkt_ineq_multipliers(
                    x_value=x_value,
                    y_value=y_sol,
                    rhs_nl=rhs_nl,
                    lb=lb,
                    ub=ub,
                )
        return {
            "status": status,
            "objective": float(m.ObjVal),
            "y": y_sol,
            "mu": mu,
            "solve_time_sec": float(getattr(m, "Runtime", 0.0)),
        }

    def _solve_nonconvex_for_x(self, x_value, solver_opts=None):
        x_value = np.asarray(x_value, dtype=float).reshape(-1)
        if x_value.shape[0] != self.n_x:
            raise ValueError("x_value must have length n_x.")

        rhs_eq = self.b + self.B @ x_value
        rhs_nl = self.beta + self.E @ x_value
        lb = self.l + self.L @ x_value
        ub = self.u + self.U @ x_value
        Qsym = 0.5 * (self.Q + self.Q.T)

        pyo, solver = self._ensure_nonconvex_solver(solver_opts=solver_opts)
        if self.nonconvex_model is None:
            self._build_nonconvex_model()
        m = self.nonconvex_model

        if self.n_eq > 0:
            for i in range(self.n_eq):
                m.eq_rhs[i] = float(rhs_eq[i])
        if self.n_ineq > 0:
            for i in range(self.n_ineq):
                m.nl_rhs[i] = float(rhs_nl[i])

        best = None
        best_obj = np.inf
        for y0 in self._initial_points(lb, ub):
            for j in range(self.n_y):
                m.y[j].setlb(float(lb[j]))
                m.y[j].setub(float(ub[j]))
                m.y[j].set_value(float(y0[j]))
            try:
                results = solver.solve(m, tee=False)
            except Exception:
                continue
            term = results.solver.termination_condition
            if term not in (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.locallyOptimal,
                pyo.TerminationCondition.feasible,
            ):
                continue
            y = np.asarray([pyo.value(m.y[j]) for j in range(self.n_y)], dtype=float)
            eq_v = np.max(np.abs(self.A @ y - rhs_eq)) if self.n_eq > 0 else 0.0
            nl_v = 0.0
            for i in range(self.n_ineq):
                val = self.a[i] @ np.exp(y) + y @ self.W[i] @ y - rhs_nl[i]
                nl_v = max(nl_v, max(val, 0.0))
            b_v = max(np.max(np.maximum(lb - y, 0.0)), np.max(np.maximum(y - ub, 0.0)))
            feas = max(eq_v, nl_v, b_v)
            oy = float(0.5 * y @ Qsym @ y + self.c @ y)
            if np.isfinite(oy) and feas <= 1e-5 and oy < best_obj:
                best_obj = oy
                best = y.copy()

        if best is None:
            return {"status": "solver_error", "objective": None, "y": None, "mu": default_ineq_multipliers(self.n_ineq)}
        return {
            "status": "optimal_inaccurate",
            "objective": float(best_obj),
            "y": best,
            "mu": default_ineq_multipliers(self.n_ineq),
        }

    def sample_parameters(self, n_samples: int):
        return _dataset_preview_points(
            x_L=self.x_L,
            x_U=self.x_U,
            num_samples=int(n_samples),
            seed=self.parameter_seed,
        )

    def get_problem_data(self):
        return {
            "n_y": self.n_y,
            "n_x": self.n_x,
            "n_eq": self.n_eq,
            "n_ineq": self.n_ineq,
            "is_diag_Q": self.is_diag_Q,
            "Q": self.Q.copy(),
            "c": self.c.copy(),
            "A": self.A.copy(),
            "b": self.b.copy(),
            "B": self.B.copy(),
            "a": self.a.copy(),
            "W": self.W.copy(),
            "beta": self.beta.copy(),
            "E": self.E.copy(),
            "l": self.l.copy(),
            "L": self.L.copy(),
            "u": self.u.copy(),
            "U": self.U.copy(),
            "x_L": self.x_L.copy(),
            "x_U": self.x_U.copy(),
            "y_feas": self.y_feas.copy(),
        }
