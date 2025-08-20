import sympy as sp
from sympy import Symbol
from typing import List


class Formulation1:
    def __init__(self, objective: str, constraints: List[str], parameters: List[str], variables: List[str]):
        self.objective_str = objective
        self.constraints_list = constraints
        self.parameters = parameters
        self.variables = variables
        self.symbol_map = {s: Symbol(s, real=True) for s in parameters + variables}
        self.objective = None
        self.equality_constraints = []
        self.inequality_constraints = []
        self.slack_syms = []
        self.lagrangian = None
        self.kkt_variables = []
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
        for c in self.constraints_list:
            expr = sp.sympify(c, evaluate=False, locals=self.symbol_map)

            if isinstance(expr, sp.Equality):
                self.equality_constraints.append(expr)
                self.eq_violation.append(expr.lhs - expr.rhs)

            elif isinstance(expr, (sp.StrictLessThan, sp.LessThan)):
                lhs = expr.lhs - expr.rhs
                self.inequality_constraints.append(lhs)
                self.ineq_violation.append(lhs)

            elif isinstance(expr, (sp.StrictGreaterThan, sp.GreaterThan)):
                lhs = expr.rhs - expr.lhs
                self.inequality_constraints.append(lhs)
                self.ineq_violation.append(lhs)

            else:
                raise ValueError(f"Unsupported constraint format: {expr}")

    def generate_lagrangian(self):
        if self.objective is None:
            self.parse_objective()
        if not self.equality_constraints and not self.inequality_constraints:
            self.parse_constraints()

        lagrangian = self.objective

        # Add equality constraints
        for i, eq in enumerate(self.equality_constraints):
            lambda_i = sp.Symbol(f"lambda_{i + 1}", real=True)
            lagrangian += lambda_i * (eq.lhs - eq.rhs)

        # Add inequality constraints with slacks
        for j, ineq in enumerate(self.inequality_constraints):
            mu_j = sp.Symbol(f"mu_{j + 1}", real=True)
            s_j = sp.Symbol(f"s_{j + 1}", real=True)
            self.symbol_map[f"s_{j + 1}"] = s_j
            self.slack_syms.append(s_j)
            lagrangian += mu_j * (ineq + s_j)

        self.lagrangian = lagrangian

    def extract_kkt_variables(self):
        all_symbols = self.lagrangian.free_symbols
        parameter_syms = {self.symbol_map[p] for p in self.parameters}
        self.kkt_variables = list(all_symbols - parameter_syms)

    def generate_kkt_system(self):
        # lambda_syms = [sp.Symbol(f"lambda_{i + 1}", real=True) for i in range(len(self.equality_constraints))]
        mu_syms = [sp.Symbol(f"mu_{j + 1}", real=True) for j in range(len(self.inequality_constraints))]
        slack_syms = self.slack_syms

        # Stationarity: ∂L/∂y = 0
        stationarity_eqns = []
        for var_name in self.variables:
            var = self.symbol_map[var_name]
            deriv = sp.diff(self.lagrangian, var)
            stationarity_eqns.append(sp.Eq(deriv, 0))

        # Primal feasibility
        primal_eqns = [sp.Eq(eq.lhs - eq.rhs, 0) for eq in self.equality_constraints]
        for ineq, s in zip(self.inequality_constraints, slack_syms):
            primal_eqns.append(sp.Eq(ineq + s, 0))

        # Complementarity: μ * s = 0
        compl_eqns = [sp.Eq(mu + s - sp.sqrt(mu ** 2 + s ** 2), 0) for mu, s in zip(mu_syms, slack_syms)]

        self.kkt_system = stationarity_eqns + primal_eqns + compl_eqns
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
