"""Train real problems and verify constraints, public reload and isolated deployment."""

from numerical_regression import run_case


def test_training_and_metadata_inference(tmp_path):
    result = run_case(tmp_path)
    assert result
