from typing import List
from KKT.formulation1 import Formulation1
from KKT.formulation2 import Formulation2
from KKT.formulation3 import Formulation3


class KKTSystemCreator:
    def __init__(self):
        self.parameters = None
        self.variables = None
        self.objective = None
        self.constraints = None
        self.model = None

    def add_parameters(self, parameters: List[str]):
        self.parameters = parameters

    def add_variables(self, variables: List[str]):
        self.variables = variables

    def add_objective(self, objective: str):
        self.objective = objective

    def add_constraints(self, constraints: List[str]):
        self.constraints = constraints

    def get_formulation1(self):
        self.model = Formulation1(self.objective, self.constraints,
                                  self.parameters, self.variables)
        self.model.formulate()

    def get_formulation2(self):
        self.model = Formulation2(self.objective, self.constraints,
                                  self.parameters, self.variables)
        self.model.formulate()

    def get_formulation3(self):
        self.model = Formulation3(self.objective, self.constraints,
                                  self.parameters, self.variables)
        self.model.formulate()

    def formulate(self, index=1):
        if index == 1:
            self.get_formulation1()
        elif index == 2:
            self.get_formulation2()
        elif index == 3:
            self.get_formulation3()

    def print_kkt_system(self):
        print("\n=== KKT System ===")
        for eq in self.model.kkt_system:
            print(eq)

    def print_kkt_variables(self):
        print("\n=== KKT Variables ===")
        print(self.model.kkt_variables)
