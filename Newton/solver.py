import sympy as sp
import numpy as np
from numpy.linalg import LinAlgError


class NewtonSolver:
    def __init__(self, kkt_system, kkt_variables, step_length=1.0, tol=1e-12, reg_factor=1e-8, max_iter=1000):
        self.kkt_system = kkt_system  # List of sp.Equality
        self.kkt_variables = kkt_variables  # List of sp.Symbol
        self.step_length = step_length
        self.tol = tol
        self.reg_factor = reg_factor
        self.max_iter = max_iter
        self.J = None  # Symbolic Jacobian
        self.residuals = None  # Residual expressions
        self.sol = None  # Final solution
        self.iterations = 0
        self.norm_list = []
        self.functional_evaluations = []
        self.create_jacobian()  # Build symbolic Jacobian

    def create_jacobian(self):
        residual_exprs = [eq.lhs - eq.rhs for eq in self.kkt_system]
        self.residuals = sp.Matrix(residual_exprs)
        self.J = self.residuals.jacobian(self.kkt_variables)

        # for eqn in self.residuals:
        #     print(eqn)

    def solve(self, initial_guess: dict, parameter_values: dict):

        # Convert variables list to ordered array for update
        xk = np.array([initial_guess[var] for var in self.kkt_variables], dtype=np.float64)

        for i in range(self.max_iter):
            # Step 1: Evaluate residual vector F(xk)
            full_subs = {**{var: val for var, val in zip(self.kkt_variables, xk)}, **parameter_values}
            Fx_evaluated = self.residuals.subs(full_subs)
            Fx_numpy = np.array(Fx_evaluated.tolist(), dtype=np.float64).flatten()

            # print(Fx_numpy)
            Fx_norm = np.linalg.norm(Fx_numpy)
            self.norm_list.append(Fx_norm)
            
            # Fx norm for convergence criteria
            if Fx_norm < self.tol:
                print(f"Converged at iteration {i - 1}")
                break

            # Step 2: Evaluate Jacobian J(xk)
            Jx_evaluated = self.J.subs(full_subs).evalf()
            Jx_numpy = np.array(Jx_evaluated.tolist(), dtype=np.float64)
            if i % 20 == 0:
                print(f"Starting Iteration {i}")

            # Step 3: Regularization if singular
            try:
                delta_x = np.linalg.solve(Jx_numpy, -Fx_numpy)
            except LinAlgError:
                print(f"WARNING: Jacobian is singular at iteration {i}; applying regularization.")
                Jx_numpy += self.reg_factor * np.eye(Jx_numpy.shape[0])
                print(Jx_numpy)
                print(np.linalg.det(Jx_numpy))
                delta_x = np.linalg.solve(Jx_numpy, -Fx_numpy)

            Fx_norm = np.linalg.norm(Fx_numpy)
            self.functional_evaluations.append(np.linalg.norm(Fx_numpy))

            # Step 5: Update xk
            xk = xk + self.step_length * delta_x
            self.iterations += 1

        if self.iterations == self.max_iter - 1 and Fx_norm > self.tol:
            print("Newton loop did not converge in {} iterations!!".format(self.max_iter))

        # Save final solution
        self.sol = {var: float(val) for var, val in zip(self.kkt_variables, xk)}
        return self.sol
