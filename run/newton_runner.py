from sympy import Symbol
import re
from KKT.creator import KKTSystemCreator


def alphanum_key(sym):
    import re
    return [int(s) if s.isdigit() else s.lower() for s in re.split('([0-9]+)', str(sym))]

def categorize(sym):
    name = str(sym)

    # Pure y variables like y1, y2
    if re.fullmatch(r"y\d+", name):
        return (0, name)

    # Other y-prefixed variables (e.g., y1x1d1)
    elif name.startswith("y") and not name.endswith("data"):
        return (1, name)

    elif name.startswith("lambda"):
        return (2, name)
    elif name.startswith("mu"):
        return (3, name)
    elif name.startswith("s"):
        return (4, name)
    elif name.startswith("delta"):
        return (5, name)
    elif name.startswith("sigma"):
        return (6, name)
    elif name.startswith("x"):
        return (7, name)
    elif name.endswith("data"):
        return (8, name)
    else:
        return (9, name)  # fallback

class NewtonRunner:
    def __init__(self, 
                 parameters, 
                 variables, 
                 objective,
                 constraints,
                 formulation_index,
                 step_length=1.0, 
                 tol=1e-12, 
                 reg_factor=1e-8, 
                 max_iter=1000):
        self.parameters = parameters
        self.variables = variables
        self.problem_variables = variables
        self.objective = objective
        self.constraints = constraints
        self.index = formulation_index
        self.step_length = step_length
        self.tol = tol
        self.reg_factor = reg_factor
        self.max_iter = max_iter
        self.creator = None
        self.solver = None
        self.create_kkt_system()
        self.formulate_kkt_system()
        # Filter only sympy.Symbols before sorting
        self.variables = sorted(
            [v for v in self.creator.model.kkt_variables if isinstance(v, Symbol)],
            key=categorize
        )

        self.parameters = sorted(
            [self.creator.model.symbol_map[p] for p in self.creator.model.parameters if isinstance(self.creator.model.symbol_map[p], Symbol)],
            key=categorize
        )

    def create_kkt_system(self):
        self.creator = KKTSystemCreator()
        self.creator.add_parameters(self.parameters)
        self.creator.add_variables(self.variables)
        self.creator.add_objective(self.objective)
        self.creator.add_constraints(self.constraints)

    def formulate_kkt_system(self):
        self.creator.formulate(self.index)
