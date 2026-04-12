import jax.numpy as jnp
from typing import NamedTuple, Optional


class LinearizationData(NamedTuple):
    A: jnp.ndarray
    b: jnp.ndarray
    C: jnp.ndarray
    d: jnp.ndarray
    h_val: jnp.ndarray
    g_val: jnp.ndarray


class QuadraticObjectiveData(NamedTuple):
    grad_f: jnp.ndarray
    Q: jnp.ndarray
    Q_diag: jnp.ndarray
    c: jnp.ndarray


class SQPSubproblemData(NamedTuple):
    objective: QuadraticObjectiveData
    constraints: LinearizationData
    l: Optional[jnp.ndarray]
    u: Optional[jnp.ndarray]


def linearize_constraints_data(eq_fun, ineq_fun, jac_eq, jac_ineq, params, y):
    A = jac_eq(y, params)
    h = eq_fun(params, y)
    b = A @ y - h

    C = jac_ineq(y, params)
    g = ineq_fun(params, y)
    d = C @ y - g

    return LinearizationData(A=A, b=b, C=C, d=d, h_val=h, g_val=g)


def linearize_constraints(eq_fun, ineq_fun, jac_eq, jac_ineq, params, y):
    data = linearize_constraints_data(
        eq_fun=eq_fun,
        ineq_fun=ineq_fun,
        jac_eq=jac_eq,
        jac_ineq=jac_ineq,
        params=params,
        y=y,
    )
    return dict(data._asdict())


def quadraticize_objective_data(
    grad_fun,
    hess_fun,
    diag_hess_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    grad = grad_fun(y, params)

    if use_diagonal_hessian:
        qdiag = diag_hess_fun(params, y)
        if diag_floor is not None:
            qdiag = jnp.maximum(qdiag, diag_floor)
        Q = jnp.diag(rho * qdiag)
    else:
        Q = rho * hess_fun(y, params)

    c = grad - Q @ y

    return QuadraticObjectiveData(grad_f=grad, Q=Q, Q_diag=jnp.diag(Q), c=c)


def quadraticize_objective(
    grad_fun,
    hess_fun,
    diag_hess_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    data = quadraticize_objective_data(
        grad_fun=grad_fun,
        hess_fun=hess_fun,
        diag_hess_fun=diag_hess_fun,
        params=params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )
    return dict(data._asdict())


def build_sqp_subproblem_data(
    eq_fun,
    ineq_fun,
    jac_eq,
    jac_ineq,
    grad_fun,
    hess_fun,
    diag_hess_fun,
    lower_fun,
    upper_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    qdata = quadraticize_objective_data(
        grad_fun=grad_fun,
        hess_fun=hess_fun,
        diag_hess_fun=diag_hess_fun,
        params=params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )

    cdata = linearize_constraints_data(
        eq_fun=eq_fun,
        ineq_fun=ineq_fun,
        jac_eq=jac_eq,
        jac_ineq=jac_ineq,
        params=params,
        y=y,
    )

    return SQPSubproblemData(
        objective=qdata,
        constraints=cdata,
        l=None if lower_fun is None else lower_fun(params),
        u=None if upper_fun is None else upper_fun(params),
    )


def build_sqp_data(
    eq_fun,
    ineq_fun,
    jac_eq,
    jac_ineq,
    grad_fun,
    hess_fun,
    diag_hess_fun,
    lower_fun,
    upper_fun,
    params,
    y,
    rho: float = 1.0,
    use_diagonal_hessian: bool = True,
    diag_floor: Optional[float] = None,
):
    data = build_sqp_subproblem_data(
        eq_fun=eq_fun,
        ineq_fun=ineq_fun,
        jac_eq=jac_eq,
        jac_ineq=jac_ineq,
        grad_fun=grad_fun,
        hess_fun=hess_fun,
        diag_hess_fun=diag_hess_fun,
        lower_fun=lower_fun,
        upper_fun=upper_fun,
        params=params,
        y=y,
        rho=rho,
        use_diagonal_hessian=use_diagonal_hessian,
        diag_floor=diag_floor,
    )
    return {
        **dict(data.objective._asdict()),
        **dict(data.constraints._asdict()),
        "l": data.l,
        "u": data.u,
    }
