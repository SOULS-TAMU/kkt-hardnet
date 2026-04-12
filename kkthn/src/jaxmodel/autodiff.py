import jax
import jax.numpy as jnp


def build_grad_y_objective(obj_flat):
    return jax.grad(lambda y, p: obj_flat(p, y), argnums=0)


def build_hess_y_objective(obj_flat):
    return jax.hessian(lambda y, p: obj_flat(p, y), argnums=0)


def build_diag_hess_y_objective(grad_fun):
    def diag_hess(params, y):
        n = y.shape[0]
        basis = jnp.eye(n, dtype=y.dtype)
        _, jvp_fun = jax.linearize(lambda yy: grad_fun(yy, params), y)
        cols = jax.vmap(jvp_fun)(basis)
        return jnp.diag(cols.T)
    return diag_hess


def build_jac_y_vector_fun(vec_fun):
    return jax.jacrev(lambda y, p: vec_fun(p, y), argnums=0)