from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxmodel import HighLevelNLPBuilder

jax.config.update("jax_enable_x64", True)

SUPPORTED_PROBLEM_TYPES = ("qp", "qcqp")
_TYPE_ALIASES = {
    "convex_qp_jaxmodel": "qp",
    "convex_qcqp_jaxmodel": "qcqp",
}


def normalize_problem_type(problem_type: str) -> str:
    normalized = str(problem_type).strip().lower()
    normalized = _TYPE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            f"Unsupported problem type '{problem_type}'. "
            f"Supported types: {', '.join(SUPPORTED_PROBLEM_TYPES)}."
        )
    return normalized


def uses_nonconvex_generator(data_cfg: Mapping[str, object]) -> bool:
    _ = data_cfg
    return False


def build_problem_generator(data_cfg: Mapping[str, object]):
    problem_type = normalize_problem_type(str(data_cfg["type"]))
    raise ValueError(
        "Legacy nonconvex "
        f"{problem_type.upper()} generation is no longer controlled by "
        "the standard poly factory. "
        "Use scripts.factory.nonconvx_factory and scripts.testcase.nonconvx_run "
        "for nonconvex datasets."
    )


def _x_abs_max(data_cfg: Mapping[str, object], dtype) -> jnp.ndarray:
    x_l = jnp.asarray(data_cfg["x_L"], dtype=dtype)
    x_u = jnp.asarray(data_cfg["x_U"], dtype=dtype)
    return jnp.maximum(jnp.abs(x_l), jnp.abs(x_u))


def _make_objective_matrix(rng: np.random.Generator, n: int, *, dtype, is_diag_q: bool) -> jnp.ndarray:
    if is_diag_q:
        diag = 1.0 + rng.uniform(0.2, 1.0, size=(n,))
        return jnp.diag(jnp.asarray(diag, dtype=dtype))

    mat = rng.normal(size=(n, n))
    spd = (mat.T @ mat) / max(1, n)
    spd = spd + np.diag(1.0 + rng.uniform(0.2, 0.8, size=(n,)))
    return jnp.asarray(spd, dtype=dtype)


def _make_objective_vector(rng: np.random.Generator, n: int, *, dtype) -> jnp.ndarray:
    return jnp.asarray(rng.uniform(-0.1, 0.1, size=(n,)), dtype=dtype)


def _make_qp_objective_vector(rng: np.random.Generator, n: int, *, dtype) -> jnp.ndarray:
    # Mirror the DC3 simple problem more closely: a positive linear term pushes
    # the equality-only optimum toward the inequality boundary.
    return jnp.asarray(0.5 + rng.random(size=(n,)), dtype=dtype)


def _min_affine_over_box(row: np.ndarray, x_l: np.ndarray, x_u: np.ndarray) -> float:
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    x_l = np.asarray(x_l, dtype=np.float64).reshape(-1)
    x_u = np.asarray(x_u, dtype=np.float64).reshape(-1)
    return float(np.sum(np.where(row >= 0.0, row * x_l, row * x_u)))


def _max_affine_over_box(row: np.ndarray, x_l: np.ndarray, x_u: np.ndarray) -> float:
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    x_l = np.asarray(x_l, dtype=np.float64).reshape(-1)
    x_u = np.asarray(x_u, dtype=np.float64).reshape(-1)
    return float(np.sum(np.where(row >= 0.0, row * x_u, row * x_l)))


def _required_active_rows(mi: int) -> int:
    return max(1, int(np.ceil(0.1 * float(mi))))


def _parameter_probe_points(
    rng: np.random.Generator,
    x_l: np.ndarray,
    x_u: np.ndarray,
    *,
    include_vertices: bool = True,
    max_vertices: int = 4096,
    max_random: int = 8,
) -> np.ndarray:
    x_l = np.asarray(x_l, dtype=np.float64).reshape(-1)
    x_u = np.asarray(x_u, dtype=np.float64).reshape(-1)
    p = int(x_l.shape[0])
    midpoint = 0.5 * (x_l + x_u)
    points = [midpoint, x_l, x_u]
    for j in range(p):
        probe_lo = midpoint.copy()
        probe_hi = midpoint.copy()
        probe_lo[j] = x_l[j]
        probe_hi[j] = x_u[j]
        points.append(probe_lo)
        points.append(probe_hi)

    if include_vertices and p > 0 and (1 << p) <= int(max_vertices):
        for mask in range(1 << p):
            vertex = x_l.copy()
            for bit in range(p):
                if (mask >> bit) & 1:
                    vertex[bit] = x_u[bit]
            points.append(vertex)

    for _ in range(min(int(max_random), max(4, 2 * p))):
        points.append(rng.uniform(x_l, x_u))

    return np.asarray(points, dtype=np.float64)


def _diagonal_quadratic_values(Y: np.ndarray, q_diag: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    Y_np = np.asarray(Y, dtype=np.float64)
    q_diag_np = np.asarray(q_diag, dtype=np.float64).reshape((-1,))
    c_vec_np = np.asarray(c_vec, dtype=np.float64).reshape((-1,))
    return 0.5 * np.sum((Y_np ** 2) * q_diag_np[None, :], axis=1) + Y_np @ c_vec_np


def _make_feasible_affine_map(
    rng: np.random.Generator,
    *,
    n: int,
    p: int,
    x_abs_max: jnp.ndarray,
    dtype,
):
    y_center = 0.15 * rng.normal(size=(n,))
    T = 0.15 * rng.normal(size=(n, p)) / max(1.0, np.sqrt(p))

    # Keep the affine feasible map numerically mild across the full parameter box.
    x_scale = np.asarray(x_abs_max, dtype=np.float64).reshape((p,))
    worst_case = np.abs(T) @ x_scale
    max_worst_case = float(np.max(worst_case)) if worst_case.size else 0.0
    if max_worst_case > 1.0:
        T *= 1.0 / max_worst_case

    return jnp.asarray(y_center, dtype=dtype), jnp.asarray(T, dtype=dtype)


def _make_equality_block(rng: np.random.Generator, me: int, n: int, p: int, *, dtype):
    if me == 0:
        return (
            jnp.zeros((0, n), dtype=dtype),
            jnp.zeros((0, p), dtype=dtype),
            jnp.zeros((0,), dtype=dtype),
        )

    # Dense, well-conditioned equality block rather than a partial identity.
    q, _ = np.linalg.qr(rng.normal(size=(n, me)))
    A = q.T
    row_scales = 0.8 + 0.4 * rng.uniform(size=(me,))
    row_signs = np.where(rng.uniform(size=(me,)) < 0.5, -1.0, 1.0)
    A = (row_scales * row_signs)[:, None] * A
    B = np.zeros((me, p), dtype=np.float64)
    b = np.zeros((me,), dtype=np.float64)
    return (
        jnp.asarray(A, dtype=dtype),
        jnp.asarray(B, dtype=dtype),
        jnp.asarray(b, dtype=dtype),
    )


def _make_affine_inequality_block(
    rng: np.random.Generator,
    mi: int,
    n: int,
    p: int,
    x_l: np.ndarray,
    x_u: np.ndarray,
    feasible_center: jnp.ndarray,
    feasible_map: jnp.ndarray,
    eq_opt_center: jnp.ndarray,
    eq_opt_map: jnp.ndarray,
    *,
    dtype,
):
    if mi == 0:
        return (
            jnp.zeros((0, n), dtype=dtype),
            jnp.zeros((0, p), dtype=dtype),
            jnp.zeros((0,), dtype=dtype),
        )

    feasible_center_np = np.asarray(feasible_center, dtype=np.float64).reshape((n,))
    feasible_map_np = np.asarray(feasible_map, dtype=np.float64).reshape((n, p))
    min_support = max(1, int(np.ceil(0.05 * float(n))))
    probe_X = _parameter_probe_points(rng, x_l, x_u, include_vertices=False)
    eq_opt_center_np = np.asarray(eq_opt_center, dtype=np.float64).reshape((n,))
    eq_opt_map_np = np.asarray(eq_opt_map, dtype=np.float64).reshape((n, p))
    probe_Y = eq_opt_center_np[None, :] + probe_X @ eq_opt_map_np.T

    best_payload = None
    best_score = -np.inf
    required_rows = _required_active_rows(mi)

    for _attempt in range(32):
        C = np.zeros((mi, n), dtype=np.float64)
        E = np.zeros((mi, p), dtype=np.float64)
        rhs_const = np.zeros((mi,), dtype=np.float64)
        for row in range(mi):
            support = rng.choice(n, size=min_support, replace=False)
            coeffs = rng.normal(size=(min_support,))
            coeff_norm = float(np.linalg.norm(coeffs))
            if coeff_norm > 1e-12:
                coeffs = coeffs / coeff_norm
            C[row, support] = coeffs

            # DC3-style constant RHS from a guaranteed-feasible particular
            # equality solution over the full parameter box.
            row_param = C[row] @ feasible_map_np
            rhs_const[row] = float(C[row] @ feasible_center_np)
            rhs_const[row] += _max_affine_over_box(row_param, x_l, x_u)

        residuals = probe_Y @ C.T - rhs_const[None, :]
        active_rows = np.unique(np.where(residuals > 1e-6)[1])
        score = float(np.max(residuals)) if residuals.size else -np.inf
        if score > best_score:
            best_score = score
            best_payload = (C.copy(), E.copy(), rhs_const.copy())
        if active_rows.size >= required_rows:
            return (
                jnp.asarray(C, dtype=dtype),
                jnp.asarray(E, dtype=dtype),
                jnp.asarray(rhs_const, dtype=dtype),
            )

    # Fallback to the most active candidate seen even if it only lights up a
    # single row on the probes.
    if best_payload is None:
        best_payload = (
            np.zeros((mi, n), dtype=np.float64),
            np.zeros((mi, p), dtype=np.float64),
            np.zeros((mi,), dtype=np.float64),
        )
    C, E, rhs_const = best_payload
    return (
        jnp.asarray(C, dtype=dtype),
        jnp.asarray(E, dtype=dtype),
        jnp.asarray(rhs_const, dtype=dtype),
    )


def _qp_equality_optimizer_map(
    Q: jnp.ndarray,
    c: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    B: jnp.ndarray,
    *,
    dtype,
):
    Q_np = np.asarray(Q, dtype=np.float64)
    c_np = np.asarray(c, dtype=np.float64).reshape((-1,))
    A_np = np.asarray(A, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64).reshape((-1,))
    B_np = np.asarray(B, dtype=np.float64)
    n = int(Q_np.shape[0])
    p = int(B_np.shape[1]) if B_np.ndim == 2 else 0
    me = int(A_np.shape[0])

    q_inv_c = np.linalg.solve(Q_np, c_np)
    if me == 0:
        return (
            jnp.asarray(-q_inv_c, dtype=dtype),
            jnp.asarray(np.zeros((n, p), dtype=np.float64), dtype=dtype),
        )

    q_inv_a_t = np.linalg.solve(Q_np, A_np.T)
    schur = A_np @ q_inv_a_t
    lam_const = np.linalg.solve(schur, b_np + A_np @ q_inv_c)
    lam_map = np.linalg.solve(schur, B_np)
    y_const = -q_inv_c + q_inv_a_t @ lam_const
    y_map = q_inv_a_t @ lam_map
    return jnp.asarray(y_const, dtype=dtype), jnp.asarray(y_map, dtype=dtype)


def _dataset_preview_points(*, x_l: np.ndarray, x_u: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    if int(num_samples) <= 0:
        return np.zeros((0, int(np.asarray(x_l, dtype=np.float64).size)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    return np.asarray(
        rng.uniform(
            np.asarray(x_l, dtype=np.float64).reshape(-1),
            np.asarray(x_u, dtype=np.float64).reshape(-1),
            size=(int(num_samples), int(np.asarray(x_l, dtype=np.float64).size)),
        ),
        dtype=np.float64,
    )


def _build_qcqp_scalar_candidate(
    row: np.ndarray,
    *,
    x_l: np.ndarray,
    x_u: np.ndarray,
    feasible_center: np.ndarray,
    feasible_map: np.ndarray,
    eq_opt_center: np.ndarray,
    eq_opt_map: np.ndarray,
    preview_X: np.ndarray,
    preview_eq_y: np.ndarray,
    support_X: np.ndarray,
    support_feas_y: np.ndarray,
    x_coeff: np.ndarray | None = None,
    rhs_margin: float = 1e-4,
):
    row_np = np.asarray(row, dtype=np.float64).reshape(-1)
    row_norm = float(np.linalg.norm(row_np))
    if row_norm <= 1e-12:
        return None
    row_np = row_np / row_norm
    p = int(np.asarray(x_l, dtype=np.float64).size)
    x_coeff_np = np.zeros((p,), dtype=np.float64) if x_coeff is None else np.asarray(x_coeff, dtype=np.float64).reshape((p,))

    feas_center_val = float(row_np @ feasible_center)
    feas_param = row_np @ feasible_map
    feas_min = feas_center_val + _min_affine_over_box(feas_param, x_l, x_u)
    feas_max = feas_center_val + _max_affine_over_box(feas_param, x_l, x_u)

    eq_center_val = float(row_np @ eq_opt_center)
    eq_param = row_np @ eq_opt_map
    eq_min = eq_center_val + _min_affine_over_box(eq_param, x_l, x_u)
    eq_max = eq_center_val + _max_affine_over_box(eq_param, x_l, x_u)

    max_abs_t = max(abs(feas_min), abs(feas_max), abs(eq_min), abs(eq_max), 1.0)
    alpha = min(0.2, 0.25 / max_abs_t)

    def _scalar_quad(val):
        val_np = np.asarray(val, dtype=np.float64)
        return 0.5 * alpha * np.square(val_np) + val_np

    if support_X.size > 0 and support_feas_y.size > 0:
        support_t = support_feas_y @ row_np
        rhs = float(np.max(_scalar_quad(support_t) - support_X @ x_coeff_np) + float(rhs_margin))
    else:
        rhs = float(max(_scalar_quad(feas_min), _scalar_quad(feas_max)) + float(rhs_margin))
    preview_t = preview_eq_y @ row_np if preview_eq_y.size else np.zeros((0,), dtype=np.float64)
    preview_residuals = _scalar_quad(preview_t) - (rhs + preview_X @ x_coeff_np if preview_X.size else rhs)
    return {
        "row": row_np,
        "Q": alpha * np.outer(row_np, row_np),
        "c": row_np.copy(),
        "rhs": rhs,
        "x_coeff": x_coeff_np,
        "preview_residuals": np.asarray(preview_residuals, dtype=np.float64),
    }


def _cluster_preview_directions(
    delta_preview: np.ndarray,
    *,
    num_clusters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    delta_np = np.asarray(delta_preview, dtype=np.float64)
    if delta_np.ndim != 2:
        raise ValueError(f"Expected a 2D preview delta matrix, received shape {delta_np.shape}.")

    num_points, n = delta_np.shape
    if num_points == 0:
        return np.zeros((0, n), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    k = max(1, min(int(num_clusters), num_points))
    centers = np.zeros((k, n), dtype=np.float64)
    first_idx = int(np.argmax(np.linalg.norm(delta_np, axis=1)))
    centers[0] = delta_np[first_idx]
    closest_dist_sq = np.sum((delta_np - centers[0]) ** 2, axis=1)

    for idx in range(1, k):
        next_idx = int(np.argmax(closest_dist_sq))
        centers[idx] = delta_np[next_idx]
        candidate_dist_sq = np.sum((delta_np - centers[idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, candidate_dist_sq)

    assignments = np.zeros((num_points,), dtype=np.int64)
    for _ in range(8):
        dist_sq = np.sum((delta_np[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dist_sq, axis=1).astype(np.int64, copy=False)
        for idx in range(k):
            mask = assignments == idx
            if np.any(mask):
                centers[idx] = np.mean(delta_np[mask], axis=0)
            else:
                centers[idx] = delta_np[int(rng.integers(num_points))]

    return centers, assignments


def _make_qcqp_quadratic_block(
    rng: np.random.Generator,
    *,
    mi: int,
    n: int,
    p: int,
    x_l: np.ndarray,
    x_u: np.ndarray,
    x_abs_max: jnp.ndarray,
    feasible_center: jnp.ndarray,
    feasible_map: jnp.ndarray,
    eq_opt_center: jnp.ndarray,
    eq_opt_map: jnp.ndarray,
    num_samples: int,
    seed: int,
    dtype,
):
    if mi == 0:
        return (
            jnp.zeros((0, n, n), dtype=dtype),
            jnp.zeros((0, n), dtype=dtype),
            jnp.zeros((0,), dtype=dtype),
            jnp.zeros((0, p), dtype=dtype),
        )

    feasible_center_np = np.asarray(feasible_center, dtype=np.float64).reshape((n,))
    feasible_map_np = np.asarray(feasible_map, dtype=np.float64).reshape((n, p))
    eq_opt_center_np = np.asarray(eq_opt_center, dtype=np.float64).reshape((n,))
    eq_opt_map_np = np.asarray(eq_opt_map, dtype=np.float64).reshape((n, p))
    preview_X = _dataset_preview_points(x_l=x_l, x_u=x_u, num_samples=num_samples, seed=seed)
    preview_eq_y = eq_opt_center_np[None, :] + preview_X @ eq_opt_map_np.T
    preview_feas_y = feasible_center_np[None, :] + preview_X @ feasible_map_np.T
    delta_preview = preview_eq_y - preview_feas_y
    activity_tol = 1e-2
    coarse_support_X = _parameter_probe_points(rng, x_l, x_u, include_vertices=True, max_random=0)
    coarse_support_feas_y = (
        feasible_center_np[None, :] + coarse_support_X @ feasible_map_np.T
        if coarse_support_X.size
        else np.zeros((0, n), dtype=np.float64)
    )
    # Guarantee the sampled dataset points retain a known feasible witness.
    support_X = (
        np.vstack([coarse_support_X, preview_X])
        if coarse_support_X.size and preview_X.size
        else (preview_X.copy() if preview_X.size else coarse_support_X)
    )
    support_feas_y = (
        np.vstack([coarse_support_feas_y, preview_feas_y])
        if coarse_support_feas_y.size and preview_feas_y.size
        else (preview_feas_y.copy() if preview_feas_y.size else coarse_support_feas_y)
    )

    preview_count = int(preview_eq_y.shape[0])
    if preview_count > 0:
        template_count = min(mi, max(4, min(preview_count, 12)))
        centers, assignments = _cluster_preview_directions(
            preview_X,
            num_clusters=template_count,
            rng=rng,
        )
        global_x_mean = np.mean(preview_X, axis=0)
        selected = []
        for cluster_idx, center_x in enumerate(centers):
            mask = assignments == cluster_idx
            if not np.any(mask):
                continue
            row = np.mean(delta_preview[mask], axis=0)
            if float(np.linalg.norm(row)) <= 1e-12:
                row = delta_preview[np.flatnonzero(mask)[0]].copy()

            raw_param_dir = np.asarray(center_x - global_x_mean, dtype=np.float64)
            param_norm = float(np.linalg.norm(raw_param_dir))
            candidate_param_dirs = [np.zeros((p,), dtype=np.float64)]
            if param_norm > 1e-12:
                unit_dir = raw_param_dir / param_norm
                for scale in (0.35, 0.7, 1.05):
                    candidate_param_dirs.append(scale * unit_dir)
                    candidate_param_dirs.append(-scale * unit_dir)

            best_candidate = None
            best_score = (-1, -np.inf, -np.inf)
            for x_coeff in candidate_param_dirs:
                candidate = _build_qcqp_scalar_candidate(
                    row,
                    x_l=x_l,
                    x_u=x_u,
                    feasible_center=feasible_center_np,
                    feasible_map=feasible_map_np,
                    eq_opt_center=eq_opt_center_np,
                    eq_opt_map=eq_opt_map_np,
                    preview_X=preview_X,
                    preview_eq_y=preview_eq_y,
                    support_X=support_X,
                    support_feas_y=support_feas_y,
                    x_coeff=x_coeff,
                )
                if candidate is None:
                    continue
                inside = candidate["preview_residuals"][mask]
                outside = candidate["preview_residuals"][~mask]
                inside_active = int(np.sum(inside > activity_tol))
                inside_median = float(np.median(inside)) if inside.size > 0 else -np.inf
                outside_high = float(np.quantile(outside, 0.75)) if outside.size > 0 else -np.inf
                score = (inside_active, inside_median - outside_high, inside_median)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is None:
                continue

            inside = best_candidate["preview_residuals"][mask]
            outside = best_candidate["preview_residuals"][~mask]
            if inside.size > 0 and outside.size > 0:
                inside_budget = float(max(np.quantile(inside, 0.4) - 1e-3, 0.0))
                outside_budget = float(max(np.quantile(outside, 0.75), 0.0))
                rhs_shift = min(inside_budget, outside_budget)
                if rhs_shift > 0.0:
                    best_candidate["rhs"] = float(best_candidate["rhs"] + rhs_shift)
                    best_candidate["preview_residuals"] = best_candidate["preview_residuals"] - rhs_shift
            selected.append(best_candidate)

        if not selected:
            selected = [
                {
                    "row": np.zeros((n,), dtype=np.float64),
                    "Q": np.zeros((n, n), dtype=np.float64),
                    "c": np.zeros((n,), dtype=np.float64),
                    "rhs": 0.0,
                    "x_coeff": np.zeros((p,), dtype=np.float64),
                    "preview_residuals": np.zeros((preview_count,), dtype=np.float64),
                }
            ]

        cover_scores = np.max(np.stack([cand["preview_residuals"] for cand in selected], axis=0), axis=0)
        uncovered = np.where(cover_scores <= activity_tol)[0]
        for idx in uncovered.tolist():
            if len(selected) >= mi:
                break
            candidate = _build_qcqp_scalar_candidate(
                delta_preview[idx],
                x_l=x_l,
                x_u=x_u,
                feasible_center=feasible_center_np,
                feasible_map=feasible_map_np,
                eq_opt_center=eq_opt_center_np,
                eq_opt_map=eq_opt_map_np,
                preview_X=preview_X,
                preview_eq_y=preview_eq_y,
                support_X=support_X,
                support_feas_y=support_feas_y,
                x_coeff=np.zeros((p,), dtype=np.float64),
            )
            if candidate is not None:
                selected.append(candidate)
    else:
        selected = [
            {
                "row": np.zeros((n,), dtype=np.float64),
                "Q": np.zeros((n, n), dtype=np.float64),
                "c": np.zeros((n,), dtype=np.float64),
                "rhs": 0.0,
                "x_coeff": np.zeros((p,), dtype=np.float64),
                "preview_residuals": np.zeros((0,), dtype=np.float64),
            }
        ]

    active_templates = list(selected)
    while len(active_templates) < mi:
        base = selected[(len(active_templates) - len(selected)) % len(selected)]
        relax_idx = len(active_templates) - len(selected)
        active_templates.append(
            {
                "row": base["row"].copy(),
                "Q": base["Q"].copy(),
                "c": base["c"].copy(),
                "rhs": float(base["rhs"] + 1.0 + 0.25 * relax_idx),
                "x_coeff": base["x_coeff"].copy(),
                "preview_residuals": base["preview_residuals"].copy(),
            }
        )

    q_best = np.stack([cand["Q"] for cand in active_templates[:mi]], axis=0)
    c_best = np.stack([cand["c"] for cand in active_templates[:mi]], axis=0)
    rhs_best = np.asarray([cand["rhs"] for cand in active_templates[:mi]], dtype=np.float64)
    e_best = np.stack([cand["x_coeff"] for cand in active_templates[:mi]], axis=0)
    return (
        jnp.asarray(q_best, dtype=dtype),
        jnp.asarray(c_best, dtype=dtype),
        jnp.asarray(rhs_best, dtype=dtype),
        jnp.asarray(e_best, dtype=dtype),
    )

def _make_affine_bounds(
    *,
    n: int,
    p: int,
    bound_radius: float,
    feasible_center: jnp.ndarray,
    feasible_map: jnp.ndarray,
    dtype,
):
    radius = float(bound_radius)
    lower_M = jnp.asarray(feasible_map, dtype=dtype).reshape((n, p))
    upper_M = jnp.asarray(feasible_map, dtype=dtype).reshape((n, p))
    center = jnp.asarray(feasible_center, dtype=dtype).reshape((n,))
    lower_c = center - radius * jnp.ones((n,), dtype=dtype)
    upper_c = center + radius * jnp.ones((n,), dtype=dtype)
    return lower_M, lower_c, upper_M, upper_c


def _build_convex_problem_data(data_cfg: Mapping[str, object], *, dtype=jnp.float64):
    problem_type = normalize_problem_type(str(data_cfg["type"]))
    p = int(data_cfg["p"])
    n = int(data_cfg["n"])
    me = int(data_cfg["me"])
    mi = int(data_cfg["mi"])

    if p <= 0 or n <= 0:
        raise ValueError("p and n must be positive.")
    if me < 0 or mi < 0:
        raise ValueError("me and mi must be nonnegative.")
    if me > n:
        raise ValueError("me cannot exceed n.")

    rng = np.random.default_rng(int(data_cfg["seed"]))
    is_diag_q = bool(data_cfg.get("is_diag_Q", False))

    Q = _make_objective_matrix(rng, n, dtype=dtype, is_diag_q=is_diag_q)
    if problem_type == "qp":
        c = _make_qp_objective_vector(rng, n, dtype=dtype)
    else:
        c = _make_objective_vector(rng, n, dtype=dtype)
    x_l = np.asarray(data_cfg["x_L"], dtype=np.float64)
    x_u = np.asarray(data_cfg["x_U"], dtype=np.float64)
    x_abs_max = _x_abs_max(data_cfg, dtype)
    A, _, _ = _make_equality_block(rng, me, n, p, dtype=dtype)
    if me > 0:
        feasible_center = jnp.asarray(0.15 * rng.normal(size=(n,)), dtype=dtype)
        rhs_map = 0.15 * rng.normal(size=(me, p)) / max(1.0, np.sqrt(p))
        rhs_worst_case = np.abs(rhs_map) @ np.asarray(x_abs_max, dtype=np.float64).reshape((p,))
        max_rhs = float(np.max(rhs_worst_case)) if rhs_worst_case.size else 0.0
        if max_rhs > 1.0:
            rhs_map *= 1.0 / max_rhs
        feasible_map = jnp.asarray(np.linalg.pinv(np.asarray(A, dtype=np.float64)) @ rhs_map, dtype=dtype)
        B = jnp.asarray(rhs_map, dtype=dtype)
        b = A @ feasible_center
    else:
        feasible_center, feasible_map = _make_feasible_affine_map(rng, n=n, p=p, x_abs_max=x_abs_max, dtype=dtype)
        B = jnp.zeros((0, p), dtype=dtype)
        b = jnp.zeros((0,), dtype=dtype)

    bound_radius = float(data_cfg.get("bound_radius", 2.0))
    lower_M, l, upper_M, u = _make_affine_bounds(
        n=n,
        p=p,
        bound_radius=bound_radius,
        feasible_center=feasible_center,
        feasible_map=feasible_map,
        dtype=dtype,
    )
    eq_opt_center, eq_opt_map = _qp_equality_optimizer_map(Q, c, A, b, B, dtype=dtype)

    payload = {
        "problem_type": problem_type,
        "n_x": p,
        "n_y": n,
        "n_eq": me,
        "n_ineq": mi,
        "is_diag_Q": bool(is_diag_q),
        "Q": np.asarray(Q, dtype=np.float64),
        "c": np.asarray(c, dtype=np.float64),
        "A": np.asarray(A, dtype=np.float64),
        "b": np.asarray(b, dtype=np.float64),
        "B": np.asarray(B, dtype=np.float64),
        "l": np.asarray(l, dtype=np.float64),
        "L": np.asarray(lower_M, dtype=np.float64),
        "u": np.asarray(u, dtype=np.float64),
        "U": np.asarray(upper_M, dtype=np.float64),
        "x_L": np.asarray(data_cfg["x_L"], dtype=np.float64),
        "x_U": np.asarray(data_cfg["x_U"], dtype=np.float64),
    }
    if problem_type == "qp":
        C, D, d = _make_affine_inequality_block(
            rng,
            mi,
            n,
            p,
            x_l,
            x_u,
            feasible_center,
            feasible_map,
            eq_opt_center,
            eq_opt_map,
            dtype=dtype,
        )
        payload["C"] = np.asarray(C, dtype=np.float64)
        payload["D"] = np.asarray(D, dtype=np.float64)
        payload["d"] = np.asarray(d, dtype=np.float64)
    else:
        q_mats, c_vecs, rhs, E = _make_qcqp_quadratic_block(
            rng,
            mi=mi,
            n=n,
            p=p,
            x_l=x_l,
            x_u=x_u,
            x_abs_max=x_abs_max,
            feasible_center=feasible_center,
            feasible_map=feasible_map,
            eq_opt_center=eq_opt_center,
            eq_opt_map=eq_opt_map,
            num_samples=int(data_cfg["num_samples"]),
            seed=int(data_cfg["seed"]),
            dtype=dtype,
        )
        payload["C"] = np.asarray(q_mats, dtype=np.float64)
        payload["d"] = np.asarray(c_vecs, dtype=np.float64)
        payload["e"] = np.asarray(rhs, dtype=np.float64)
        payload["E"] = np.asarray(E, dtype=np.float64)

    return payload


def build_problem_data(data_cfg: Mapping[str, object]):
    return _build_convex_problem_data(data_cfg)


def build_problem_model_from_data(problem_data: Mapping[str, object], *, dtype=jnp.float64):
    problem_type = normalize_problem_type(str(problem_data["problem_type"]))
    p = int(problem_data["n_x"])
    n = int(problem_data["n_y"])
    me = int(problem_data["n_eq"])
    mi = int(problem_data["n_ineq"])

    Q = jnp.asarray(problem_data["Q"], dtype=dtype)
    c = jnp.asarray(problem_data["c"], dtype=dtype)
    A = jnp.asarray(problem_data["A"], dtype=dtype)
    b = jnp.asarray(problem_data["b"], dtype=dtype)
    B = jnp.asarray(problem_data["B"], dtype=dtype)
    lower_M = jnp.asarray(problem_data["L"], dtype=dtype)
    l = jnp.asarray(problem_data["l"], dtype=dtype)
    upper_M = jnp.asarray(problem_data["U"], dtype=dtype)
    u = jnp.asarray(problem_data["u"], dtype=dtype)
    params = {"x": jnp.zeros((p,), dtype=dtype)}

    builder = (
        HighLevelNLPBuilder(dtype=dtype)
        .add_variable("y", n)
        .add_parameter("x", p)
        .set_quadratic_objective(Q=Q, c=c)
    )

    if me > 0:
        builder = builder.add_affine_equality(
            var_name="y",
            A=A,
            rhs_const=b,
            param_terms=[(B, "x")],
            name="eq_block",
        )

    if problem_type == "qp" and mi > 0:
        C = jnp.asarray(problem_data["C"], dtype=dtype)
        D = jnp.asarray(problem_data["D"], dtype=dtype)
        d = jnp.asarray(problem_data["d"], dtype=dtype)
        builder = builder.add_affine_inequality(
            var_name="y",
            C=C,
            rhs_const=d,
            param_terms=[(D, "x")],
            name="ineq_block",
        )
    elif problem_type == "qcqp" and mi > 0:
        quad_Q = jnp.asarray(problem_data["C"], dtype=dtype)
        quad_c = jnp.asarray(problem_data["d"], dtype=dtype)
        quad_rhs = jnp.asarray(problem_data["e"], dtype=dtype)
        quad_x_coeff = jnp.asarray(problem_data["E"], dtype=dtype)
        for idx in range(mi):
            builder = builder.add_quadratic_inequality(
                Q=quad_Q[idx],
                c=quad_c[idx],
                rhs_const=float(quad_rhs[idx]),
                x_coeff=quad_x_coeff[idx],
                x_name="x",
                name=f"qc_{idx}",
            )

    return (
        builder
        .set_affine_lower_bound(var_name="y", param_name="x", M=lower_M, c=l)
        .set_affine_upper_bound(var_name="y", param_name="x", M=upper_M, c=u)
        .build(example_params=params, jit_compile=True)
    )


def build_problem_model(data_cfg: Mapping[str, object], *, dtype=jnp.float64):
    problem_data = _build_convex_problem_data(data_cfg)
    return build_problem_model_from_data(problem_data, dtype=dtype)
