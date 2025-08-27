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

        # ===============================
        # Define the neural network
        # ===============================
        self.nn = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims),
            nn.ReLU(),
            nn.Linear(self.hidden_dims, len(variables)),  # exclude differential vars
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
