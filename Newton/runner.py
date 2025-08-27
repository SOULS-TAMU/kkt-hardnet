import numpy as np
from typing import List
from KKT.creator import KKTSystemCreator
from Newton.solver import NewtonSolver
import json
import numpy as np
from pathlib import Path
from Newton.runner import NewtonRunner

def run_from_directory(config_dir):
    config_dir = Path(config_dir)

    # === Load configs ===
    with open(config_dir / "problem.json", "r") as f:
        P = json.load(f)

    with open(config_dir / "parameters.json", "r") as f:
        param_dict = json.load(f)

    with open(config_dir / "initialization.json", "r") as f:
        init_dict = json.load(f)

    with open(config_dir / "config.json", "r") as f:
        config_dict = json.load(f)

    # === Extract model info ===
    objective = P['objective']
    constraints = P['constraints']
    parameters = P['parameters']
    variables = P['variables']
    binary_variables = P['binary_variables']

    # === Convert ordered values ===
    parameter_values = [param_dict[p] for p in parameters]
    initial_guess = [init_dict[v] for v in variables + binary_variables]

    # === Run Newton solver ===
    runner = NewtonRunner(
        parameters, variables, binary_variables,
        objective, constraints,
        parameter_values=parameter_values,
        formulation_index=config_dict["formulation_index"],
        step_length=config_dict["step_length"],
        tol=config_dict["tol"],
        reg_factor=config_dict["reg_factor"],
        max_iter=config_dict["max_iter"]
    )

    runner.run(initial_guess)

    solution = runner.get_solution()
    metrics = runner.get_metrics()

    # === Save results ===
    solution_serializable = {str(k): float(v) for k, v in solution.items()}
    metrics_serializable = {
        str(k): [float(val) for val in v] if isinstance(v, list) else float(v)
        for k, v in metrics.items()
    }

    # Save as JSON
    with open(config_dir / "solution.json", "w") as f_sol:
        json.dump(solution_serializable, f_sol, indent=4)

    with open(config_dir / "metrics.json", "w") as f_met:
        json.dump(metrics_serializable, f_met, indent=4)



class NewtonRunner:
    def __init__(self, parameters, variables, binary_variables, objective, constraints,
                 formulation_index, parameter_values: List,
                 step_length=1.0, tol=1e-12, reg_factor=1e-8, max_iter=1000):
        self.norm_list = None
        self.iterations = None
        self.solution = None
        self.parameters = parameters
        self.variables = variables
        self.binary_variables = binary_variables
        self.problem_variables = variables + binary_variables
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
        # self.variables = self.creator.model.kkt_variables
        self.variables = sorted(self.creator.model.kkt_variables, key=lambda sym: sym.name)
        # print(self.variables)
        self.initial_guess = {}
        self.parameter_dict = {self.creator.model.symbol_map[param]:
                                   param_val for param, param_val in
                               zip(self.creator.model.parameters,
                                   parameter_values)}

    def create_kkt_system(self):
        self.creator = KKTSystemCreator()
        self.creator.add_parameters(self.parameters)
        self.creator.add_variables(self.variables)
        self.creator.add_objective(self.objective)
        self.creator.add_constraints(self.constraints)

    def formulate_kkt_system(self):
        self.creator.formulate(self.index)

    def solve_kkt_system(self):
        self.solver = NewtonSolver(self.creator.model.kkt_system,
                                   self.variables,
                                   step_length=self.step_length,
                                   tol=self.tol,
                                   reg_factor=self.reg_factor,
                                   max_iter=self.max_iter)

        self.solution = self.solver.solve(self.initial_guess,
                                          self.parameter_dict)

    def run(self, initial_guess):
        """
        Initialize KKT variables using:
        - Provided initial values for problem variables (like y1, y2)
        - Random normal initialization for all other KKT variables
        """
        # Step 1: Map initial guesses to problem variables
        for var, val in zip(self.creator.model.variables, initial_guess):
            self.initial_guess[self.creator.model.symbol_map[var]] = val
        # self.initial_guess = {
        #     var: val for self.creator.model.symbol_map(var), val in zip(self.creator.model.variables, initial_guess)
        # }
        # print(self.creator.model.symbol_map)
        # Step 2: Add random initialization for remaining KKT variables
        for var in self.variables:
            if var not in self.initial_guess.keys():
                # np.random.seed(42)
                self.initial_guess[var] = np.random.rand()
        # print(self.initial_guess)
        self.solve_kkt_system()
        self.extract_solver_metrics()

    def extract_solver_metrics(self):
        self.iterations = self.solver.iterations
        self.norm_list = self.solver.norm_list

    def get_solution(self, value_tolerance=1e-12):
        """
    Returns solution dictionary with values rounded to 12 decimals
    and thresholded (abs(val) < tolerance → 0.0).
    """
        cleaned_solution = {}
        for var, val in self.solution.items():
            if abs(val) < value_tolerance:
                cleaned_solution[var] = 0.0
            else:
                cleaned_solution[var] = round(val, 12)
        return cleaned_solution

    def get_metrics(self, value_tolerance=1e-12):
        """
    Returns metrics with log-norm values, thresholded similarly.
    """
        norm_log = []
        for norm in self.norm_list:
            val = np.log10(norm) if norm > 0 else float('-inf')  # handle log(0)
            if abs(val) < value_tolerance:
                norm_log.append(0.0)
            else:
                norm_log.append(round(val, 12))
        return {
            "iterations": self.iterations,
            "Norm": norm_log
        }
