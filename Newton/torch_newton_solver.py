import torch
import sympy as sp
import numpy as np


class SymbolicNewtonSolver:
    def __init__(self, equations, variables, tol=1e-8, max_iter=100, step_size=1.0, reg=1e-8):
        """
        Args:
            equations: list of sympy expressions (e.g., [f1(x), f2(x)])
            variables: list of sympy symbols (e.g., [x, y])
        """
        self.equations = equations
        self.variables = variables
        self.tol = tol
        self.max_iter = max_iter
        self.step_size = step_size
        self.reg = reg

        # Symbolic Residual Vector
        self.residual_vec = sp.Matrix(equations)
        # Symbolic Jacobian
        self.jacobian_sym = self.residual_vec.jacobian(variables)

        # Lambdify for evaluation
        self.f_func = sp.lambdify(variables, self.residual_vec, modules='torch')
        self.j_func = sp.lambdify(variables, self.jacobian_sym, modules='torch')

    def solve(self, initial_guess):
        """
        Args:
            initial_guess: dict or list of initial values for the variables
        Returns:
            solution: dictionary of variable -> value
        """
        # Initialize x_k as a torch tensor
        if isinstance(initial_guess, dict):
            xk = torch.tensor([initial_guess[v] for v in self.variables], dtype=torch.float32)
        else:
            xk = torch.tensor(initial_guess, dtype=torch.float32)

        xk.requires_grad_(False)

        for i in range(self.max_iter):
            # Evaluate residual and Jacobian using torch
            F = torch.tensor(self.f_func(*xk), dtype=torch.float32).flatten()
            print(F)
            J = torch.tensor(self.j_func(*xk), dtype=torch.float32)
            print(J)

            # Regularize if Jacobian is near-singular
            try:
                delta_x = torch.linalg.solve(J + self.reg * torch.eye(J.shape[0]), -F)
            except RuntimeError:
                print(f"Jacobian is singular at iteration {i}")
                break

            # Update
            xk_new = xk + self.step_size * delta_x

            # Check for convergence
            if torch.norm(delta_x).item() < self.tol:
                print(f"Converged in {i + 1} iterations.")
                return {str(var): float(val) for var, val in zip(self.variables, xk_new)}

            xk = xk_new

        print("Newton solver did not converge within max_iter.")
        return {str(var): float(val) for var, val in zip(self.variables, xk)}


# Define variables and equations
x, y = sp.symbols('x y')
eqns = [
    x ** 2 + y ** 2 - 4,
    x - y
]

variables = [x, y]
initial_guess = {x: 2.0, y: 1.9}

# Create and solve
solver = SymbolicNewtonSolver(eqns, variables)
solution = solver.solve(initial_guess)

print("Solution:")
for k, v in solution.items():
    print(f"{k} = {v:.6f}")
