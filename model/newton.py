import torch
import re
import sympy as sp
from torch import nn


class NewtonLayer(nn.Module):
    def __init__(self, residuals, eq_violation, ineq_violation, variables, parameters, step_length=0.1, tol=1e-12, reg_factor=1e-6, max_iter=100):
        super(NewtonLayer, self).__init__()
        self.variables = variables
        self.parameters = parameters
        self.res_exprs = residuals
        self.J_exprs = residuals.jacobian(variables)
        
        self.eq_viol_exprs = eq_violation
        self.ineq_viol_exprs = ineq_violation
        self.step_size = step_length
        self.tol = tol
        self.reg_factor = reg_factor
        self.max_iter = max_iter

        def categorize(sym):
            name = str(sym)

            # Pure y variables like y1, y2
            if re.fullmatch(r"y\d+", name):
                return (0, name)

            # Derivative variables: dy1dx1 or d1y1dx1 or d2y3dx4
            elif re.fullmatch(r"d\d*y\d+dx\d+", name) or re.fullmatch(r"dy\d+dx\d+", name):
                return (1, name)

            # Other y-prefixed variables (e.g., y1x1d1)
            elif name.startswith("y") and not name.endswith("data"):
                return (2, name)

            elif name.startswith("lambda"):
                return (3, name)
            elif name.startswith("mu"):
                return (4, name)
            elif name.startswith("s"):
                return (5, name)
            elif name.startswith("delta"):
                return (6, name)
            elif name.startswith("sigma"):
                return (7, name)
            elif name.startswith("x"):
                return (8, name)
            elif name.endswith("data"):
                return (9, name)
            else:
                return (10, name)  # fallback

        # Combine all symbols and sort them
        self.all_syms = sorted(variables + parameters, key=categorize)
        self.sym_names = [str(s) for s in self.all_syms]
        # print("Symbol Names: ", self.sym_names)

        # Build differentiable torch-compatible functions
        self.res_fn = self._make_torch_fn(self.res_exprs, is_matrix=False)
        self.jac_fn = self._make_torch_fn(self.J_exprs.tolist(), is_matrix=True)
        
        
        if len(self.eq_viol_exprs):
            self.eq_viol_fn = self._make_torch_fn(self.eq_viol_exprs, is_matrix=False)
        else:
            self.eq_viol_fn = None
        
        if len(self.ineq_viol_exprs):
            self.ineq_viol_fn = self._make_torch_fn(self.ineq_viol_exprs, is_matrix=False)
        else:
            self.ineq_viol_fn = None
    
    def _evaluate_eq_res(self, y_data, x_data):
        bsz, device, dtype = y_data.shape[0], y_data.device, y_data.dtype
        if self.eq_viol_fn == None:
            return torch.zeros(bsz, 0, device=device, dtype=dtype)
        else:
            return self.eq_viol_fn(y_data, x_data)
        
    def _evaluate_ineq_res(self, y_data, x_data):
        bsz, device, dtype = y_data.shape[0], y_data.device, y_data.dtype
        if self.ineq_viol_fn == None:
            return torch.zeros(bsz, 0, device=device, dtype=dtype)
        else:
            return self.ineq_viol_fn(y_data, x_data)
        
        
    def _make_torch_fn(self, sym_exprs, is_matrix=False):
        torch_exprs = []

        # Build lambdify expressions
        if is_matrix:
            for row in sym_exprs:
                row_exprs = []
                for expr in row:
                    f = sp.lambdify(self.all_syms, expr, modules=torch)
                    row_exprs.append(f)
                torch_exprs.append(row_exprs)
        else:
            for expr in sym_exprs:
                f = sp.lambdify(self.all_syms, expr, modules=torch)
                torch_exprs.append(f)

        def torch_fn(y, x):
            # Build dict {symbol_name: column tensor}
            input_dict = {}

            for i, sym in enumerate(self.variables):
                input_dict[str(sym)] = y[:, i]

            for i, sym in enumerate(self.parameters):
                input_dict[str(sym)] = x[:, i]

            # Assemble input in correct symbolic order
            inputs = [input_dict[name] for name in self.sym_names]

            if is_matrix:
                rows = []
                for row_expr in torch_exprs:
                    row = []
                    for f in row_expr:
                        val = f(*inputs)
                        if not isinstance(val, torch.Tensor):
                            val = torch.full((y.shape[0],), float(val), device=y.device)
                        row.append(val)
                    rows.append(torch.stack(row, dim=1))
                return torch.stack(rows, dim=1)
            else:
                vals = []
                for f in torch_exprs:
                    val = f(*inputs)
                    if not isinstance(val, torch.Tensor):
                        val = torch.full((y.shape[0],), float(val), device=y.device)
                    vals.append(val)
                return torch.stack(vals, dim=1)

        return torch_fn

    def forward(self, y, x):
        yk = y.clone()  # (B, n)
        # print("Intial Y: ")
        # print(yk)
        B, n = yk.shape

        for iter in range(self.max_iter):
            r = self.res_fn(yk, x)  # (B, n)
            J = self.jac_fn(yk, x)  # (B, n, n)
            # print("Residual Functional Value: ")
            # print(r)
            # print("Jacobian Functional Value: ")
            # print(J)

            # print("Residual: ")
            # print(r)
            # print("Jacobian: ")
            # print(J)

            norm = torch.linalg.norm(r, dim=1)  # (B,)
            # if torch.all(norm < self.tol):
            #     print(f"Newton Loop Converged at iteration {iter}!!")
            #     break

            try:
                delta_x = torch.linalg.solve(J, -r)  # (B, n)
                
            except RuntimeError:
                # print("Jacobian is Singular!")
                JT = J.transpose(-2,-1)
                JTJ = JT @ J
                I = torch.eye(JTJ.size(-1))
                A = JTJ + self.reg_factor * I.unsqueeze(0)
                b = JT @ (-r.unsqueeze(-1))
                delta_x   = torch.linalg.solve(A, b).squeeze(-1) 
                
                # I = torch.eye(n, device=yk.device).expand(B, n, n)
                # J = J + self.reg_factor * I
                # delta_x = torch.linalg.solve(J, -r)
                
                # print(delta_x)

            # print(yk.shape)
            # print(self.delta_x)
            yk = yk + self.step_size * delta_x

            # print("Current Y: ")
            # print(yk)

        # if iter == self.max_iter - 1 and torch.any(norm > self.tol):
            # print("Newton loop did not converge in {} iterations!!".format(self.max_iter))
        return yk
