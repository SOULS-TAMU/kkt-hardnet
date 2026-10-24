"""JAX-only deployment facade."""

from pathlib import Path
from ohagi.inference import export, load
from kkthn.api.artifacts import kkthn_graph
from purin.solvers import ProjectionMethod


def export_predictor(self, *, method=None, directory=None, platforms=None):
    """Export the trained KKTHN model for metadata-only single-instance JAX inference."""
    selected = (
        ProjectionMethod.normalize(method).value
        if method is not None
        else self._last_method
    )
    if selected is None or selected not in self._trained:
        raise RuntimeError("Train or load the selected model before exporting it.")
    state = self._trained[selected]
    root = (
        Path(directory).expanduser().resolve()
        if directory is not None
        else Path(state.result["output_dir"]) / "export"
    )
    info = export(
        root / "metadata.json",
        **kkthn_graph(self, state=state),
        platforms=("cpu",) if platforms is None else tuple(platforms),
    )
    return {
        "metadata_path": info["metadata_path"],
        "predictor": load(info["metadata_path"]),
        "manifest_path": info["metadata_path"],
        "portable_path": str(root / info["portable"]["file"]),
    }
