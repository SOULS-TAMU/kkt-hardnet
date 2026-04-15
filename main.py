from __future__ import annotations

from pathlib import Path

from kkthn import KKTHardNet


TRAIN = {
    "epochs": 1200,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "train_frac": 0.8,
    "hidden_size": 64,
    "hidden_layers": 2,
    "seed": 42,
    "dtype": "float64",
    "print_every": 1,
    "newton_step_length": 0.5,
    "newton_tol": 1e-6,
    "newton_reg_factor": 1e-3,
    "max_newton_iter": 30,
    "max_backtrack_iter": 10,
}


def build_model(param_path: str | Path, var_path: str | Path | None = None) -> KKTHardNet:
    model = KKTHardNet(name="kkt_hardnet", train=TRAIN)
    x = model.add_parameter(["x1", "x2"])
    theta = model.add_inverse_parameter(["a0", "a1"], init_value=[10.0, -10.0])
    y = model.add_variable(["y1", "y2", "y3"])

    model.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
    model.constraints.add(
        theta.a0 * y.y1 + y.y2 - x.x1 == 0,
        y.y2 - theta.a1 * y.y3 - x.x2 == 0,
        y.y1**2 + y.y3**2 <= 2.0,
        y.y1 >= 0,
    )
    model.dataset(parameters=param_path, variables=var_path)
    return model


if __name__ == "__main__":
    parameters = Path("parameters.csv")
    variables = Path("variables.csv")

    model = build_model(parameters, variables)
    result = model.model()
    print(f"Metadata saved to: {result['metadata_path']}")
