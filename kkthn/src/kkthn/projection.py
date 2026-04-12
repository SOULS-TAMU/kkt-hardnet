from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ProjectionSettings:
    fb_eps: float = 1e-8
    gn_max_iters: int = 25
    gn_tol: float = 1e-8
    gn_reg: float = 1e-6
    armijo_alpha: float = 1e-4
    armijo_beta: float = 0.5
    armijo_max_steps: int = 15
    backward_reg: float = 1e-8


@dataclass(frozen=True)
class ProjectionLayer:
    project: Callable
    project_single: Callable
    solve_single: Callable
    eq_constraints: Callable
    ineq_constraints: Callable
    dims: dict[str, int]
    settings: ProjectionSettings


def _fb(mu, s, eps: float):
    return jnp.sqrt(mu * mu + s * s + eps * eps) - mu - s


def _param_dim(model, param_name: str) -> int:
    if model.parameter_spec is None:
        raise ValueError("JaxNLPModel must define a parameter spec.")
    if param_name not in model.parameter_spec.shapes:
        raise KeyError(f"Parameter '{param_name}' is not present in the model.")
    return int(np.prod(model.parameter_spec.shapes[param_name]))


def make_projection_layer(
    model,
    *,
    param_name: str = "x",
    settings: ProjectionSettings | None = None,
) -> ProjectionLayer:
    """Create the KKT-HardNet projection layer from a JaxNLPModel.

    The layer solves, per sample,

        min_y 0.5 ||y - y_hat||^2
        s.t.  h(y, x) = 0, g(y, x) <= 0, lb(x) <= y <= ub(x)

    Inequalities and bounds are folded into one slack/complementarity block.
    """

    cfg = settings or ProjectionSettings()
    ny = int(model.var_spec.total_size)
    nx = _param_dim(model, param_name)
    x0 = jnp.zeros((nx,), dtype=model.dtype)
    y0 = jnp.zeros((ny,), dtype=model.dtype)
    params0 = {param_name: x0}
    ne = int(model.eq_residual(params0, y0).shape[0])
    raw_ni = int(model.ineq_residual(params0, y0).shape[0])
    has_lb = model.bounds.lower_fun is not None
    has_ub = model.bounds.upper_fun is not None
    ni = raw_ni + (ny if has_lb else 0) + (ny if has_ub else 0)
    nz = ny + ne + ni + ni

    def make_params(x):
        return {param_name: x}

    def eq_fun(y, x):
        return model.eq_residual(make_params(x), y)

    def ineq_fun(y, x):
        params = make_params(x)
        parts = []
        base = model.ineq_residual(params, y)
        if raw_ni > 0:
            parts.append(base)
        if has_lb:
            parts.append(model.lower_bounds(params) - y)
        if has_ub:
            parts.append(y - model.upper_bounds(params))
        if not parts:
            return jnp.zeros((0,), dtype=y.dtype)
        return jnp.concatenate(parts, axis=0)

    def pack_z(y, lam, mu, s):
        return jnp.concatenate([y, lam, mu, s], axis=0)

    def unpack_z(z):
        y = z[:ny]
        lam = z[ny : ny + ne]
        mu = z[ny + ne : ny + ne + ni]
        s = z[ny + ne + ni : ny + ne + 2 * ni]
        return y, lam, mu, s

    def init_z(x, y_hat):
        y = y_hat
        lam = jnp.zeros((ne,), dtype=y_hat.dtype)
        mu = jnp.ones((ni,), dtype=y_hat.dtype) * 1e-2
        gi0 = ineq_fun(y, x)
        s = jnp.maximum(-gi0, 1e-3) if ni > 0 else jnp.zeros((0,), dtype=y_hat.dtype)
        return pack_z(y, lam, mu, s)

    def _kkt_residual(z, x, y_hat):
        y, lam, mu, s = unpack_z(z)

        def lagrangian(yy):
            obj = 0.5 * jnp.dot(yy - y_hat, yy - y_hat)
            eq_term = jnp.dot(lam, eq_fun(yy, x)) if ne > 0 else jnp.asarray(0.0, dtype=yy.dtype)
            ineq_term = jnp.dot(mu, ineq_fun(yy, x)) if ni > 0 else jnp.asarray(0.0, dtype=yy.dtype)
            return obj + eq_term + ineq_term

        grad_y = jax.grad(lagrangian)(y)
        ce = eq_fun(y, x)
        gi = ineq_fun(y, x)
        comp = _fb(mu, s, cfg.fb_eps)
        return jnp.concatenate([grad_y, ce, gi + s, comp], axis=0)

    kkt_residual = jax.jit(_kkt_residual)
    kkt_jac_z = jax.jit(jax.jacobian(_kkt_residual, argnums=0))
    kkt_jac_x = jax.jit(jax.jacobian(_kkt_residual, argnums=1))
    kkt_jac_yhat = jax.jit(jax.jacobian(_kkt_residual, argnums=2))

    @jax.jit
    def merit(z, x, y_hat):
        r = kkt_residual(z, x, y_hat)
        return 0.5 * jnp.dot(r, r)

    @jax.jit
    def gn_direction(z, x, y_hat):
        r = kkt_residual(z, x, y_hat)
        J = kkt_jac_z(z, x, y_hat)
        JTJ = J.T @ J + cfg.gn_reg * jnp.eye(nz, dtype=z.dtype)
        rhs = -(J.T @ r)
        return jnp.linalg.solve(JTJ, rhs), r

    @jax.jit
    def armijo_line_search(z, d, x, y_hat):
        phi0 = merit(z, x, y_hat)
        r0 = kkt_residual(z, x, y_hat)
        J0 = kkt_jac_z(z, x, y_hat)
        grad_phi_dot_d = jnp.dot(J0.T @ r0, d)

        def body_fun(state, _):
            step, accepted = state
            phi_trial = merit(z + step * d, x, y_hat)
            ok = phi_trial <= phi0 + cfg.armijo_alpha * step * grad_phi_dot_d
            new_step = jnp.where(accepted, step, jnp.where(ok, step, cfg.armijo_beta * step))
            return (new_step, jnp.logical_or(accepted, ok)), None

        (step, _), _ = lax.scan(
            body_fun,
            init=(jnp.asarray(1.0, dtype=z.dtype), jnp.asarray(False)),
            xs=None,
            length=cfg.armijo_max_steps,
        )
        return step

    @jax.jit
    def solve_single(x, y_hat):
        z0 = init_z(x, y_hat)
        r0 = kkt_residual(z0, x, y_hat)
        res0 = jnp.linalg.norm(r0, ord=2)

        def cond_fun(state):
            k, _z, res_norm = state
            return jnp.logical_and(k < cfg.gn_max_iters, res_norm > cfg.gn_tol)

        def body_fun(state):
            k, z, _res_norm = state
            d, _ = gn_direction(z, x, y_hat)
            step = armijo_line_search(z, d, x, y_hat)
            z_new = z + step * d
            r_new = kkt_residual(z_new, x, y_hat)
            return k + 1, z_new, jnp.linalg.norm(r_new, ord=2)

        return lax.while_loop(cond_fun, body_fun, (0, z0, res0))

    @jax.custom_vjp
    def project_single(x, y_hat):
        _iters, z_star, _res = solve_single(x, y_hat)
        y_star, _lam, _mu, _s = unpack_z(z_star)
        return y_star

    def project_single_fwd(x, y_hat):
        iters, z_star, res_norm = solve_single(x, y_hat)
        y_star, _lam, _mu, _s = unpack_z(z_star)
        return y_star, (z_star, x, y_hat, iters, res_norm)

    def project_single_bwd(res, g_y):
        z_star, x, y_hat, _iters, _res_norm = res
        Jz = kkt_jac_z(z_star, x, y_hat)
        rhs = jnp.concatenate([g_y, jnp.zeros((ne + ni + ni,), dtype=g_y.dtype)], axis=0)
        reg = cfg.backward_reg * jnp.eye(nz, dtype=Jz.dtype)
        v = jnp.linalg.solve(Jz.T + reg, rhs)
        Jx = kkt_jac_x(z_star, x, y_hat)
        Jyhat = kkt_jac_yhat(z_star, x, y_hat)
        g_x = -(Jx.T @ v)
        g_yhat = -(Jyhat.T @ v)
        return g_x, g_yhat

    project_single.defvjp(project_single_fwd, project_single_bwd)
    project = jax.jit(jax.vmap(project_single, in_axes=(0, 0)))

    @jax.jit
    def eq_constraints(y, x):
        return eq_fun(y, x)

    @jax.jit
    def ineq_constraints(y, x):
        return ineq_fun(y, x)

    return ProjectionLayer(
        project=project,
        project_single=project_single,
        solve_single=solve_single,
        eq_constraints=eq_constraints,
        ineq_constraints=ineq_constraints,
        dims={"n_x": nx, "n_y": ny, "n_eq": ne, "n_ineq": ni, "n_z": nz, "raw_n_ineq": raw_ni},
        settings=cfg,
    )
