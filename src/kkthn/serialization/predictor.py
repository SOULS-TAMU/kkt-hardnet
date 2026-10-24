"""Frontend identity for the shared OHAGI JAX predictor writer."""

from ohagi.inference import Predictor as Predictor
from ohagi.serialization.predictor import (
    make_network_apply as make_network_apply,
    load_predictor as load_predictor,
)
from ohagi.serialization.predictor import save_predictor as _save


def save_predictor(
    directory,
    *,
    network_spec,
    params,
    input_size,
    output_size,
    batch_size,
    projection_spec=None,
    problem=None,
    input_transform=None,
    dtype="float64",
    backend="cpu",
    metadata=None,
    predictor_factory=None,
    factory_config=None,
    _inference_spec=None,
):
    """Save a prepared KKTHN predictor using the shared OHAGI JAX format."""
    options = dict(locals())
    if isinstance(predictor_factory, str) and ":" not in predictor_factory:
        from kkthn.serialization.registry import resolve

        predictor_factory = resolve(predictor_factory)
    options["predictor_factory"] = predictor_factory
    return _save(framework="kkthn", **options)
