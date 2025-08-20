import re
import torch.nn as nn
from model.newton import NewtonLayer


class NewtonModel(nn.Module):
    def __init__(self, residuals, eq_violation, ineq_violation, variables, parameters, input_dim, config):
        super(NewtonModel, self).__init__()
        self.hidden_dims = config["hidden_dim"]
        self.step_length = config["newton_step_length"]
        self.tol = config["newton_tol"]
        self.reg_factor = config["newton_reg_factor"]
        self.max_newton_iter = config["max_newton_iter"]

        # print(variables)

        # ===============================
        # Separate differential and non-differential variables
        # ===============================
        self.symbolic_vars = variables + parameters
        self.required_derivatives = []
        self.has_differential_terms = False
        self.max_diff_order = 0

        self.diff_variable_names = []
        self.non_diff_variable_names = []

        for name in variables:
            name = str(name)
            # Match dy1dx1 (first-order)
            if re.fullmatch(r"dy\d+dx\d+", name):
                self.has_differential_terms = True
                self.diff_variable_names.append(name)
                y_idx, x_idx = re.findall(r"\d+", name)
                self.required_derivatives.append({
                    'target': f'y{y_idx}',
                    'order': 1,
                    'wrt': [f'x{x_idx}'],
                    'symbol': name
                })
                self.max_diff_order = max(self.max_diff_order, 1)

            # Match higher-order e.g. d2y3dx1dx2
            elif match := re.fullmatch(r"d(\d+)y(\d+)dx(\d+)(dx\d+)*", name):
                self.has_differential_terms = True
                order = int(match.group(1))
                y_idx = int(match.group(2))
                dx_terms = re.findall(r"dx(\d+)", name)
                self.diff_variable_names.append(name)
                self.required_derivatives.append({
                    'target': f'y{y_idx}',
                    'order': order,
                    'wrt': [f'x{i}' for i in dx_terms],
                    'symbol': name
                })
                self.max_diff_order = max(self.max_diff_order, order)

            else:
                self.non_diff_variable_names.append(name)

        self.num_diff_terms = len(self.diff_variable_names)

        # ===============================
        # Define the neural network (only for non-differential vars)
        # ===============================
        self.nn = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, len(self.non_diff_variable_names)),  # exclude differential vars
            # nn.ReLU()
        )

        # ===============================
        # Define the NewtonLayer
        # ===============================
        self.newton = NewtonLayer(
            residuals=residuals,
            eq_violation = eq_violation,
            ineq_violation= ineq_violation,
            variables=variables,
            parameters=parameters,
            step_length=self.step_length,
            tol=self.tol,
            reg_factor=self.reg_factor,
            max_iter=self.max_newton_iter
        )

    def forward(self, x):
        # Only predict non-differential vars
        y_nn = self.nn(x)  # (B, len(non_diff_variable_names))

        # You are expected to reconstruct full y vector here using autograd
        # Including manually computing required gradients and concatenating with y_nn

        # Call Newton layer with full vector
        y_tilde = self.newton(y_nn, x)
        return y_tilde
