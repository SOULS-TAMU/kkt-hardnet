import sympy as sp
from sympy import Symbol, Eq
from typing import List


class Formulation2:
    def __init__(self, objective: str, constraints: List[str], parameters: List[str],
                 variables: List[str]):
        self.kkt_variables = []
        self.objective_str = objective
        self.constraints_list = constraints
        self.parameters = parameters
        self.variables = variables

        # Create symbol map
        self.symbol_map = {s: Symbol(s, real=True) for s in parameters + variables}
        self.objective = None
        self.equality_constraints = []   # all constraints are converted to equalities
        self.aux_delta_vars = []         # store delta_j symbols
        self.lagrangian = None
        self.kkt_system = []
        self.residuals = []
        self.eq_violation = []
        self.ineq_violation = []

    def _build_auto_objective(self):
        y_like_names = [name for name in self.variables if name.startswith("y")]
        terms = []
        for yname in y_like_names:
            pdata = f"{yname}_data"
            if pdata not in self.parameters:
                self.parameters.append(pdata)
                self.symbol_map[pdata] = Symbol(pdata, real=True)
            y = self.symbol_map[yname]
            y_data = self.symbol_map[pdata]
            terms.append((y - y_data)**2)

        if not terms:
            # Fallback to a constant zero objective if no y-variables exist
            return sp.Integer(0)

        return sp.Rational(1, 2) * sp.Add(*terms)

    def parse_objective(self):
        # If user passes "auto" (or None / ""), synthesize the objective
        if self.objective_str in (None, "", "auto"):
            self.objective = self._build_auto_objective()
        else:
            self.objective = sp.sympify(self.objective_str, evaluate=False, locals=self.symbol_map)

            y_like_names = [name for name in self.variables if name.startswith("y")]
        
            for yname in y_like_names:
                pdata = f"{yname}_data"
                if pdata not in self.parameters:
                    self.parameters.append(pdata)
                    self.symbol_map[pdata] = Symbol(pdata, real=True)

    def parse_constraints(self):
        for j, c in enumerate(self.constraints_list):
            expr = sp.sympify(c, evaluate=False, locals=self.symbol_map)

            if isinstance(expr, sp.Equality):
                self.equality_constraints.append(expr)
                self.eq_violation.append(expr.lhs - expr.rhs)

            elif isinstance(expr, (sp.StrictLessThan, sp.LessThan)):
                g_expr = expr.lhs - expr.rhs
                delta_j = Symbol(f"delta_{j+1}", real=True)
                self.aux_delta_vars.append(delta_j)
                eq = Eq(g_expr + delta_j**2, 0)  # g(y) + δ² = 0
                self.equality_constraints.append(eq)
                self.symbol_map[str(delta_j)] = delta_j
                self.ineq_violation.append(g_expr)

            elif isinstance(expr, (sp.StrictGreaterThan, sp.GreaterThan)):
                g_expr = expr.rhs - expr.lhs
                delta_j = Symbol(f"delta_{j+1}", real=True)
                self.aux_delta_vars.append(delta_j)
                eq = Eq(g_expr + delta_j**2, 0)  # g(y) + δ² = 0
                self.equality_constraints.append(eq)
                self.symbol_map[str(delta_j)] = delta_j
                self.ineq_violation.append(g_expr)

            else:
                raise ValueError(f"Unsupported constraint format: {expr}")

    def generate_lagrangian(self):
        self.lagrangian = self.objective
        for i, eq in enumerate(self.equality_constraints):
            lambda_i = sp.Symbol(f"lambda_{i+1}", real=True)
            self.lagrangian += lambda_i * (eq.lhs - eq.rhs)

    def extract_kkt_variables(self):
        # All free symbols in the lagrangian
        all_symbols = self.lagrangian.free_symbols

        # Convert parameter names to actual sympy symbols
        parameter_syms = {self.symbol_map[p] for p in self.parameters}

        # Exclude parameters to get KKT variables
        self.kkt_variables = list(all_symbols - parameter_syms)

    def generate_kkt_system(self):
        # lambda_syms = [sp.Symbol(f"lambda_{i+1}", real=True) for i in range(len(self.equality_constraints))]
        kkt_eqns = []

        # Stationarity: ∇_y L = 0 for all primal vars (including deltas)
        all_vars = self.variables + [str(d) for d in self.aux_delta_vars]
        for var_name in all_vars:
            var = self.symbol_map[var_name]
            deriv = sp.diff(self.lagrangian, var)
            kkt_eqns.append(sp.Eq(deriv, 0))

        # Primal feasibility: all constraints should be satisfied
        for eq in self.equality_constraints:
            kkt_eqns.append(sp.Eq(eq.lhs - eq.rhs, 0))

        self.kkt_system = kkt_eqns

        for eq in self.kkt_system:
            self.residuals.append(eq.lhs)
        self.residuals = sp.Matrix(self.residuals)

    def formulate(self):
        self.parse_objective()
        self.parse_constraints()
        self.generate_lagrangian()
        self.extract_kkt_variables()
        self.generate_kkt_system()
        # print("Eq_Violation:", self.eq_violation)
        # print("Ineq_Violation:", self.ineq_violation)
        # print(self.objective)
        # print(self.parameters)
