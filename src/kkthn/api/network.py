"""Neural-network initialization and primal/dual output handling."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from ohagi.nn.outputs import split_pretraining_outputs
from ohagi.nn.outputs import split_projection_outputs
from purin.problems.expressions import _EvaluationContext
from ohagi.training import Adam
from kkthn.api.configuration import _dtype


def _network_setup(self, config, X_train):
    dtype = _dtype(config["dtype"])
    backbone = self.nn.build()
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    if not config["standardize_inputs"]:
        mean = np.zeros_like(mean)
        scale = np.ones_like(scale)
    n_eq = sum(item.kind == "eq" for item in self.constraints)
    n_ineq = sum(item.kind == "ineq" for item in self.constraints)
    output_size = self.nn.output_size(
        n_y=len(self.variable_names), n_eq=n_eq, n_ineq=n_ineq
    )
    if n_ineq and float(config["dual_pretrain_gamma"]) <= float(
        self.nn.inequality_dual_floor
    ):
        raise ValueError(
            "dual_pretrain_gamma must exceed inequality_dual_floor when "
            "inequality duals are predicted."
        )
    parameters = backbone.init(
        jax.random.PRNGKey(int(config["seed"])),
        input_dim=len(self.parameter_names),
        output_dim=output_size,
        dtype=dtype,
        n_primal_outputs=(len(self.variable_names) if self.nn.predict_duals else None),
        n_equality_duals=(n_eq if self.nn.predict_duals else 0),
        n_inequality_duals=(n_ineq if self.nn.predict_duals else 0),
        dual_initial_gamma=float(config["dual_pretrain_gamma"]),
        inequality_dual_floor=float(self.nn.inequality_dual_floor),
    )
    optimizer = Adam(learning_rate=float(config["learning_rate"]))
    return backbone, parameters, optimizer, mean, scale, dtype


def _network_primal_bounds(self, X, dtype):
    """Infer box bounds from simple affine one-variable inequalities.

    Every symbolic inequality is normalized by the problem frontend to
    ``g(theta, y) <= 0``. A row is treated as a primal box bound only when its
    residual is affine in the variables and depends on exactly one decision
    variable. For

    ``a(theta) * y_j + b(theta) <= 0``

    ``a < 0`` contributes a lower bound and ``a > 0`` contributes an upper
    bound. Multi-variable and nonlinear inequalities remain projection-only
    constraints and are deliberately ignored here.
    """

    parameter_values = jnp.asarray(X, dtype=dtype)
    squeeze = False
    if parameter_values.ndim == 1:
        parameter_values = parameter_values[None, :]
        squeeze = True
    if parameter_values.ndim != 2:
        raise ValueError(
            "Network parameter batches must have shape (batch, n_parameters)."
        )

    n_variables = len(self.variable_names)
    batch_size = parameter_values.shape[0]
    lower = jnp.full((batch_size, n_variables), -jnp.inf, dtype=dtype)
    upper = jnp.full((batch_size, n_variables), jnp.inf, dtype=dtype)

    variable_index = {name: index for index, name in enumerate(self.variable_names)}
    parameter_index = {name: index for index, name in enumerate(self.parameter_names)}
    zero_variables = jnp.zeros((n_variables,), dtype=dtype)
    coefficient_tolerance = jnp.asarray(1e-12, dtype=dtype)

    for constraint in self.constraints:
        if constraint.kind != "ineq":
            continue
        residual = constraint.residual
        if not residual.affine_in_variables:
            continue
        if residual.affine_coefficients is None or residual.affine_offset is None:
            continue
        dependencies = tuple(residual.variable_dependencies)
        if len(dependencies) != 1:
            continue

        variable_name = dependencies[0]
        variable_position = variable_index[variable_name]
        coefficient_expression = residual.affine_coefficients.get(variable_name)
        if coefficient_expression is None:
            continue
        offset_expression = residual.affine_offset

        def evaluate_row(theta):
            context = _EvaluationContext(
                variables=zero_variables,
                parameters=theta,
                variable_index=variable_index,
                parameter_index=parameter_index,
            )
            coefficient = jnp.asarray(
                coefficient_expression.evaluate(context), dtype=dtype
            )
            offset = jnp.asarray(offset_expression.evaluate(context), dtype=dtype)
            nonzero = jnp.abs(coefficient) > coefficient_tolerance
            safe_coefficient = jnp.where(
                nonzero, coefficient, jnp.asarray(1.0, dtype=dtype)
            )
            bound = -offset / safe_coefficient
            lower_candidate = jnp.where(
                coefficient < -coefficient_tolerance, bound, -jnp.inf
            )
            upper_candidate = jnp.where(
                coefficient > coefficient_tolerance, bound, jnp.inf
            )
            return lower_candidate, upper_candidate

        lower_candidate, upper_candidate = jax.vmap(evaluate_row)(parameter_values)
        lower = lower.at[:, variable_position].set(
            jnp.maximum(lower[:, variable_position], lower_candidate)
        )
        upper = upper.at[:, variable_position].set(
            jnp.minimum(upper[:, variable_position], upper_candidate)
        )

    if squeeze:
        return lower[0], upper[0]
    return lower, upper


def _network_outputs(self, backbone, parameters, X, mean, scale, dtype):
    values = backbone.apply(
        parameters,
        (X - jnp.asarray(mean, dtype=dtype)) / jnp.asarray(scale, dtype=dtype),
    )
    n_eq = sum(item.kind == "eq" for item in self.constraints)
    n_ineq = sum(item.kind == "ineq" for item in self.constraints)
    primal_lower, primal_upper = _network_primal_bounds(self, X, dtype)
    return split_projection_outputs(
        values,
        n_y=len(self.variable_names),
        n_eq=n_eq,
        n_ineq=n_ineq,
        predict_duals=self.nn.predict_duals,
        inequality_dual_floor=float(self.nn.inequality_dual_floor),
        primal_lower_bounds=primal_lower,
        primal_upper_bounds=primal_upper,
        # Custom override
        primal_transform_fn=getattr(self, "primal_transform_fn", None),
        problem_parameters=X,
    )


def _network_pretraining_outputs(
    self, backbone, parameters, X, mean, scale, dtype, gamma
):
    """Evaluate the raw network with boxed primal and latent-dual outputs."""

    values = backbone.apply(
        parameters,
        (X - jnp.asarray(mean, dtype=dtype)) / jnp.asarray(scale, dtype=dtype),
    )
    n_eq = sum(item.kind == "eq" for item in self.constraints)
    n_ineq = sum(item.kind == "ineq" for item in self.constraints)
    primal_lower, primal_upper = _network_primal_bounds(self, X, dtype)
    return split_pretraining_outputs(
        values,
        n_y=len(self.variable_names),
        n_eq=n_eq,
        n_ineq=n_ineq,
        predict_duals=self.nn.predict_duals,
        dual_pretrain_gamma=float(gamma),
        primal_lower_bounds=primal_lower,
        primal_upper_bounds=primal_upper,
    )


def _network_primal_output(
    self, backbone, parameters, X, mean, scale, dtype, gamma=1e-3
):
    """Return the visible boxed primal head during raw pretraining."""
    y, _lambda_boxed, _mu_boxed = self._network_pretraining_outputs(
        backbone, parameters, X, mean, scale, dtype, gamma
    )
    return y
