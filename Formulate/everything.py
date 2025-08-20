import sympy as sp
import numpy as np
from sympy import Symbol, Eq
from typing import List
from numpy.linalg import LinAlgError
from KKT.creator import KKTSystemCreator

class Res_Var_Par:
    def __init__(self, parameters, variables, binary_variables, objective, constraints,
                 formulation_index):
        self.parameters = parameters
        self.variables = variables
        self.binary_variables = binary_variables
        self.problem_variables = variables + binary_variables
        self.objective = objective
        self.constraints = constraints
        self.index = formulation_index
        self.creator = None
        self.residuals = None
        self.create_kkt_system()
        self.formulate_kkt_system()
        # self.variables = self.creator.model.kkt_variables
        self.variables = sorted(self.creator.model.kkt_variables, key=lambda sym: sym.name)
        self.parameters = sorted(self.creator.model.kkt_parameters, key=lambda sym: sym.name)
        self.residuals = self.creator.model.residuals
        
        
    def create_kkt_system(self):
        self.creator = KKTSystemCreator()
        self.creator.add_parameters(self.parameters)
        self.creator.add_variables(self.variables)
        self.creator.add_binary_variables(self.binary_variables)
        self.creator.add_objective(self.objective)
        self.creator.add_constraints(self.constraints)

    def formulate_kkt_system(self):
        self.creator.formulate(self.index)
    
