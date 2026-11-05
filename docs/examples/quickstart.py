import jax

jax.config.update("jax_enable_x64", True)
from pathlib import Path
import numpy as np
from kkthn import KKTHardNet

p_values = np.linspace(0.4, 1.2, 8)
data = Path("data.csv")
np.savetxt(
    data,
    np.column_stack((p_values, 0.6 * p_values, 0.4 * p_values)),
    delimiter=",",
    header="p,u,v",
    comments="",
)

model = KKTHardNet(
    name="sum_model",
    train={
        "epochs": 2,
        "batch_size": 2,
        "seed": 7,
        "pretrain": False,
        "print_every": 0,
    },
)
p = model.add_parameter(["p"])
y = model.add_variable(["u", "v"])
model.constraints.add(y.u + y.v == p.p, y.u >= 0, y.v >= 0)
model.dataset(parameters=data, variables=data)
result = model.model()
prediction = model.predict([0.7])

# Check the declared physical/output contract and model transport.

np.testing.assert_allclose(np.sum(prediction), 0.7, atol=1e-6)
assert np.min(prediction) >= -1e-6
restored = KKTHardNet().load(result["metadata_path"])
np.testing.assert_allclose(restored.predict([0.7]), prediction, atol=1e-7)
model.export_inference("deployment/metadata.json")
from ohagi.inference import load

np.testing.assert_allclose(
    load("deployment/metadata.json").predict([0.7]), prediction, atol=1e-7
)
