"""Measure actual trained frontend metadata inference; timings are informational."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc


def measure(fn, x, jax):
    for _ in range(10):
        jax.block_until_ready(fn(x))
    values = []
    for _ in range(150):
        start = time.perf_counter_ns()
        jax.block_until_ready(fn(x))
        values.append((time.perf_counter_ns() - start) / 1000)
    return {
        "median_us": statistics.median(values),
        "p95_us": sorted(values)[142],
        "calls": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, default=Path("performance-results.json"))
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "src/kkthn/api").is_dir():
        args.output.write_text(
            json.dumps(
                {"available": False, "reason": "Base predates shared-engine frontend"}
            )
            + "\n"
        )
        return
    sys.path.insert(0, str(source / "src"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numerical_regression import run_case
    from ohagi.inference import load

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        run_case(directory)
        training = time.perf_counter() - start
        path = Path(directory) / "metadata.json"
        start = time.perf_counter()
        predictor = load(path)
        preparation = time.perf_counter() - start
        x = np.array([0.7])
        expected = predictor.predict(x)
        np.testing.assert_allclose(
            predictor.predict_jax(jnp.asarray(x)), expected, atol=1e-10
        )
        metrics = {
            "loaded_device_single": measure(predictor.predict_jax, jnp.asarray(x), jax),
            "loaded_host_single": measure(predictor.predict, x, jax),
        }
        tracemalloc.start()
        predictor.predict(x)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    report = {
        "available": True,
        "package": "kkthn",
        "source": str(source),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "workload": "Seeded small trained CPU frontend, not an MPC/application benchmark",
        "training_and_validation_seconds": training,
        "load_and_warmup_seconds": preparation,
        "python_peak_single_call_bytes": peak,
        "metrics": metrics,
    }
    if args.compare:
        base = json.loads(args.compare.read_text())
        report["comparison_available"] = base.get("available", False)
        if base.get("available"):
            report["median_ratios"] = {
                k: metrics[k]["median_us"] / base["metrics"][k]["median_us"]
                for k in metrics
            }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(
                "## Informational inference performance\n\n```json\n"
                + json.dumps(report, indent=2)
                + "\n```\n"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
