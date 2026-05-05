from __future__ import annotations

import ast
import ctypes
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_FORMAT = "kkthn-native-projection-v2"
_SOURCE_VERSION = "kkt-newton-20260505"


class _CExpr(ast.NodeVisitor):
    def __init__(self, parameter_names: list[str], variable_names: list[str]) -> None:
        self.parameter_index = {name: idx for idx, name in enumerate(parameter_names)}
        self.variable_index = {name: idx for idx, name in enumerate(variable_names)}

    def visit_Expression(self, node: ast.Expression) -> str:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> str:
        if node.id in self.parameter_index:
            return f"x[{self.parameter_index[node.id]}]"
        if node.id in self.variable_index:
            return f"y[{self.variable_index[node.id]}]"
        raise ValueError(f"Unsupported symbol in native projection expression: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> str:
        if isinstance(node.value, (int, float)):
            return repr(float(node.value))
        raise ValueError(f"Unsupported constant in native projection expression: {node.value!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        raise ValueError("Unsupported unary operator in native projection expression.")

    def visit_BinOp(self, node: ast.BinOp) -> str:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return f"({lhs} + {rhs})"
        if isinstance(node.op, ast.Sub):
            return f"({lhs} - {rhs})"
        if isinstance(node.op, ast.Mult):
            return f"({lhs} * {rhs})"
        if isinstance(node.op, ast.Div):
            return f"({lhs} / {rhs})"
        if isinstance(node.op, ast.Pow):
            return f"pow({lhs}, {rhs})"
        raise ValueError("Unsupported binary operator in native projection expression.")

    def visit_Call(self, node: ast.Call) -> str:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call in native projection expression.")
        allowed = {"sin", "cos", "exp", "log", "sqrt", "abs"}
        name = node.func.id
        if name not in allowed or len(node.args) != 1:
            raise ValueError(f"Unsupported function in native projection expression: {name}")
        fn = "fabs" if name == "abs" else name
        return f"{fn}({self.visit(node.args[0])})"

    def generic_visit(self, node) -> str:
        raise ValueError(f"Unsupported syntax in native projection expression: {type(node).__name__}")


def _compile_expr(expr: str, parameter_names: list[str], variable_names: list[str]) -> str:
    tree = ast.parse(expr, mode="eval")
    return _CExpr(parameter_names, variable_names).visit(tree)


def _parameter_names(problem: dict[str, Any]) -> list[str]:
    return [str(name) for name in problem.get("parameters", [])] + [
        str(name) for name in problem.get("inverse_parameters", [])
    ]


def _source(problem: dict[str, Any], settings: dict[str, Any]) -> str:
    parameters = _parameter_names(problem)
    variables = [str(name) for name in problem.get("variables", [])]
    eq_exprs = [
        _compile_expr(item["residual"], parameters, variables)
        for item in problem.get("constraint_details", [])
        if item.get("kind") == "eq"
    ]
    ineq_exprs = [
        _compile_expr(item["residual"], parameters, variables)
        for item in problem.get("constraint_details", [])
        if item.get("kind") == "ineq"
    ]
    nx = len(parameters)
    ny = len(variables)
    ne = len(eq_exprs)
    ni = len(ineq_exprs)
    nz = ny + ne + ni + ni
    max_iter = int(settings.get("gn_max_iters", settings.get("max_newton_iter", 30)))
    tol = float(settings.get("gn_tol", settings.get("newton_tol", 1e-6)))
    reg = float(settings.get("gn_reg", settings.get("newton_reg_factor", 1e-3)))
    step_length = float(settings.get("newton_step_length", 0.5))
    backtrack = int(settings.get("max_backtrack_iter", settings.get("armijo_max_steps", 10)))
    fb_eps = float(settings.get("fb_eps", 1e-8))

    eq_body = "\n".join(f"    out[{idx}] = {expr};" for idx, expr in enumerate(eq_exprs)) or "    (void)out;"
    ineq_body = "\n".join(f"    out[{idx}] = {expr};" for idx, expr in enumerate(ineq_exprs)) or "    (void)out;"

    return f"""
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define KKTHN_EXPORT __declspec(dllexport)
#else
#define KKTHN_EXPORT
#endif

#define NX {nx}
#define NY {ny}
#define NE {ne}
#define NI {ni}
#define NZ {nz}
#define MAX_ITER {max_iter}
#define TOL {tol:.17g}
#define REG {reg:.17g}
#define STEP_LENGTH {step_length:.17g}
#define BACKTRACK {backtrack}
#define FB_EPS {fb_eps:.17g}

static double max2(double a, double b) {{ return a > b ? a : b; }}

static void eval_eq(const double *x, const double *y, double *out) {{
{eq_body}
    (void)x; (void)y;
}}

static void eval_ineq(const double *x, const double *y, double *out) {{
{ineq_body}
    (void)x; (void)y;
}}

static void unpack(const double *z, double *y, double *lam, double *mu, double *s) {{
    for (int i = 0; i < NY; ++i) y[i] = z[i];
    for (int i = 0; i < NE; ++i) lam[i] = z[NY + i];
    for (int i = 0; i < NI; ++i) mu[i] = z[NY + NE + i];
    for (int i = 0; i < NI; ++i) s[i] = z[NY + NE + NI + i];
}}

static double lagrangian(const double *x, const double *y, const double *yhat, const double *lam, const double *mu) {{
    double ce[NE > 0 ? NE : 1];
    double gi[NI > 0 ? NI : 1];
    eval_eq(x, y, ce);
    eval_ineq(x, y, gi);
    double val = 0.0;
    for (int i = 0; i < NY; ++i) {{
        double d = y[i] - yhat[i];
        val += 0.5 * d * d;
    }}
    for (int i = 0; i < NE; ++i) val += lam[i] * ce[i];
    for (int i = 0; i < NI; ++i) val += mu[i] * gi[i];
    return val;
}}

static void residual(const double *x, const double *yhat, const double *z, double *r) {{
    double y[NY > 0 ? NY : 1], lam[NE > 0 ? NE : 1], mu[NI > 0 ? NI : 1], s[NI > 0 ? NI : 1];
    double ce[NE > 0 ? NE : 1], gi[NI > 0 ? NI : 1];
    unpack(z, y, lam, mu, s);
    double eps = 1e-6;
    for (int j = 0; j < NY; ++j) {{
        double yp[NY > 0 ? NY : 1];
        double ym[NY > 0 ? NY : 1];
        for (int k = 0; k < NY; ++k) {{ yp[k] = y[k]; ym[k] = y[k]; }}
        yp[j] += eps;
        ym[j] -= eps;
        r[j] = (lagrangian(x, yp, yhat, lam, mu) - lagrangian(x, ym, yhat, lam, mu)) / (2.0 * eps);
    }}
    eval_eq(x, y, ce);
    eval_ineq(x, y, gi);
    for (int i = 0; i < NE; ++i) r[NY + i] = ce[i];
    for (int i = 0; i < NI; ++i) r[NY + NE + i] = gi[i] + s[i];
    for (int i = 0; i < NI; ++i) {{
        double a = mu[i], b = s[i];
        r[NY + NE + NI + i] = sqrt(a * a + b * b + FB_EPS * FB_EPS) - a - b;
    }}
}}

static double merit(const double *x, const double *yhat, const double *z) {{
    double r[NZ > 0 ? NZ : 1];
    residual(x, yhat, z, r);
    double out = 0.0;
    for (int i = 0; i < NZ; ++i) out += r[i] * r[i];
    return 0.5 * out;
}}

static int solve_linear(double *A, double *b, double *x, int n) {{
    for (int i = 0; i < n; ++i) {{
        int piv = i;
        double best = fabs(A[i * n + i]);
        for (int r = i + 1; r < n; ++r) {{
            double v = fabs(A[r * n + i]);
            if (v > best) {{ best = v; piv = r; }}
        }}
        if (best < 1e-14) return 1;
        if (piv != i) {{
            for (int c = i; c < n; ++c) {{
                double tmp = A[i * n + c]; A[i * n + c] = A[piv * n + c]; A[piv * n + c] = tmp;
            }}
            double tb = b[i]; b[i] = b[piv]; b[piv] = tb;
        }}
        double diag = A[i * n + i];
        for (int r = i + 1; r < n; ++r) {{
            double f = A[r * n + i] / diag;
            A[r * n + i] = 0.0;
            for (int c = i + 1; c < n; ++c) A[r * n + c] -= f * A[i * n + c];
            b[r] -= f * b[i];
        }}
    }}
    for (int i = n - 1; i >= 0; --i) {{
        double acc = b[i];
        for (int c = i + 1; c < n; ++c) acc -= A[i * n + c] * x[c];
        x[i] = acc / A[i * n + i];
    }}
    return 0;
}}

static int solve_one(const double *x, const double *yhat, double *yout) {{
    double z[NZ > 0 ? NZ : 1];
    for (int i = 0; i < NY; ++i) z[i] = yhat[i];
    for (int i = 0; i < NE; ++i) z[NY + i] = 0.0;
    for (int i = 0; i < NI; ++i) z[NY + NE + i] = 1e-2;
    if (NI > 0) {{
        double gi[NI > 0 ? NI : 1];
        eval_ineq(x, yhat, gi);
        for (int i = 0; i < NI; ++i) z[NY + NE + NI + i] = max2(-gi[i], 1e-3);
    }}

    for (int iter = 0; iter < MAX_ITER; ++iter) {{
        double r[NZ > 0 ? NZ : 1];
        residual(x, yhat, z, r);
        double rn = 0.0;
        for (int i = 0; i < NZ; ++i) rn += r[i] * r[i];
        rn = sqrt(rn);
        if (rn <= TOL) break;

        double J[NZ * NZ > 0 ? NZ * NZ : 1];
        double zp[NZ > 0 ? NZ : 1], zm[NZ > 0 ? NZ : 1], rp[NZ > 0 ? NZ : 1], rm[NZ > 0 ? NZ : 1];
        for (int c = 0; c < NZ; ++c) {{
            for (int k = 0; k < NZ; ++k) {{ zp[k] = z[k]; zm[k] = z[k]; }}
            double h = 1e-6 * (1.0 + fabs(z[c]));
            zp[c] += h;
            zm[c] -= h;
            residual(x, yhat, zp, rp);
            residual(x, yhat, zm, rm);
            for (int rr = 0; rr < NZ; ++rr) J[rr * NZ + c] = (rp[rr] - rm[rr]) / (2.0 * h);
        }}

        double A[NZ * NZ > 0 ? NZ * NZ : 1];
        double b[NZ > 0 ? NZ : 1];
        double d[NZ > 0 ? NZ : 1];
        for (int i = 0; i < NZ; ++i) {{
            b[i] = 0.0;
            d[i] = 0.0;
            for (int j = 0; j < NZ; ++j) A[i * NZ + j] = (i == j ? REG : 0.0);
        }}
        for (int rr = 0; rr < NZ; ++rr) {{
            for (int c = 0; c < NZ; ++c) {{
                b[c] -= J[rr * NZ + c] * r[rr];
                for (int k = 0; k < NZ; ++k) A[c * NZ + k] += J[rr * NZ + c] * J[rr * NZ + k];
            }}
        }}
        if (solve_linear(A, b, d, NZ) != 0) break;

        double phi0 = merit(x, yhat, z);
        double step = STEP_LENGTH;
        double trial[NZ > 0 ? NZ : 1];
        int accepted = 0;
        for (int bt = 0; bt <= BACKTRACK; ++bt) {{
            for (int k = 0; k < NZ; ++k) trial[k] = z[k] + step * d[k];
            if (merit(x, yhat, trial) <= phi0 || bt == BACKTRACK) {{
                for (int k = 0; k < NZ; ++k) z[k] = trial[k];
                accepted = 1;
                break;
            }}
            step *= 0.5;
        }}
        if (!accepted) break;
    }}
    for (int i = 0; i < NY; ++i) yout[i] = z[i];
    return 0;
}}

KKTHN_EXPORT int kkthn_project_batch(int batch, const double *x, const double *yhat, double *yout) {{
    for (int b = 0; b < batch; ++b) {{
        int rc = solve_one(x + b * NX, yhat + b * NY, yout + b * NY);
        if (rc != 0) return rc;
    }}
    return 0;
}}
"""


class NativeProjection:
    def __init__(self, shared_path: str | Path, *, n_x: int, n_y: int) -> None:
        self.shared_path = Path(shared_path)
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.lib = ctypes.CDLL(str(self.shared_path))
        self.lib.kkthn_project_batch.argtypes = [
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        ]
        self.lib.kkthn_project_batch.restype = ctypes.c_int

    def project(self, x: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1, self.n_x))
        y_arr = np.ascontiguousarray(np.asarray(y_hat, dtype=np.float64).reshape(-1, self.n_y))
        out = np.empty_like(y_arr)
        rc = self.lib.kkthn_project_batch(int(x_arr.shape[0]), x_arr, y_arr, out)
        if rc != 0:
            raise RuntimeError(f"Native projection failed with code {rc}.")
        return out


def _platform_tag() -> str:
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return f"{system}-{machine}"


def _shared_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "projection_native.dll"
    if system == "darwin":
        return "projection_native.dylib"
    return "projection_native.so"


def _native_cache_root() -> Path:
    override = os.environ.get("KKTHN_NATIVE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "windows":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(Path.home())
        return Path(base) / "kkthn" / "native_projection"
    if system == "darwin":
        return Path.home() / "Library" / "Caches" / "kkthn" / "native_projection"
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "kkthn" / "native_projection"
    return Path.home() / ".cache" / "kkthn" / "native_projection"


def _native_cache_dir() -> Path:
    return _native_cache_root() / _SOURCE_VERSION / _platform_tag()


def _native_artifact_dir(run_dir: str | Path, *, platform_tag: str | None = None) -> Path:
    tag = str(platform_tag or _platform_tag())
    return Path(run_dir) / "native_projection" / _SOURCE_VERSION / tag


def _native_library_path(run_dir: str | Path, *, platform_tag: str | None = None) -> Path:
    return _native_artifact_dir(run_dir, platform_tag=platform_tag) / _shared_name()


def _compiler_candidates(explicit: str | None = None) -> list[str]:
    if explicit is not None:
        return [explicit]
    candidates: list[str] = []
    env_cc = os.environ.get("CC")
    if env_cc:
        candidates.append(env_cc)
    if platform.system().lower() == "windows":
        candidates.extend(["cl", "gcc", "clang"])
    else:
        candidates.extend(["cc", "gcc", "clang"])
    out: list[str] = []
    for candidate in candidates:
        if candidate in out:
            continue
        if Path(candidate).is_absolute() or shutil.which(candidate) is not None:
            out.append(candidate)
    return out


def _compile_command(compiler: str, source: Path, shared: Path) -> list[str]:
    system = platform.system().lower()
    compiler_name = Path(compiler).name.lower()
    if system == "windows" and compiler_name in {"cl", "cl.exe"}:
        return [compiler, "/O2", "/LD", str(source), f"/Fe:{shared}"]
    if system == "darwin":
        return [compiler, "-O3", "-std=c99", "-fPIC", "-dynamiclib", str(source), "-lm", "-o", str(shared)]
    return [compiler, "-O3", "-std=c99", "-fPIC", "-shared", str(source), "-lm", "-o", str(shared)]


def _platform_payload(
    *,
    source: str | None = None,
    shared_library: str | None = None,
    status: str,
    compiler: str | None = None,
    command: list[str] | None = None,
    returncode: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    n_x: int | None = None,
    n_y: int | None = None,
) -> dict[str, Any]:
    return {
        "source_version": _SOURCE_VERSION,
        "source": source,
        "source_stored": False,
        "shared_library": shared_library,
        "platform": platform.system(),
        "machine": platform.machine(),
        "platform_tag": _platform_tag(),
        "compiler": compiler,
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "status": status,
        "n_x": n_x,
        "n_y": n_y,
    }


def _manifest_payload(
    *,
    current: dict[str, Any],
    artifacts: dict[str, dict[str, Any]] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    platform_tag = str(current.get("platform_tag", _platform_tag()))
    stored_artifacts = {
        str(tag): dict(entry)
        for tag, entry in (artifacts or {}).items()
        if isinstance(entry, dict)
    }
    stored_artifacts[platform_tag] = dict(current)
    payload = {
        "format": _FORMAT,
        "source_version": _SOURCE_VERSION,
        "source": current.get("source"),
        "source_stored": bool(current.get("source_stored", False)),
        "shared_library": current.get("shared_library"),
        "cache_scope": "run",
        "platform": current.get("platform"),
        "machine": current.get("machine"),
        "platform_tag": platform_tag,
        "batch_mode": "sequential",
        "compiler": current.get("compiler"),
        "command": current.get("command"),
        "returncode": current.get("returncode"),
        "stdout": current.get("stdout"),
        "stderr": current.get("stderr"),
        "status": current.get("status"),
        "n_x": current.get("n_x"),
        "n_y": current.get("n_y"),
        "artifacts": stored_artifacts,
    }
    if message is not None:
        payload["message"] = message
    return payload


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _artifact_entries(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for tag, entry in artifacts.items():
            if not isinstance(entry, dict):
                continue
            normalized = dict(entry)
            normalized.setdefault("platform_tag", str(tag))
            normalized.setdefault("source_version", payload.get("source_version", _SOURCE_VERSION))
            out[str(normalized["platform_tag"])] = normalized
    legacy_tag = payload.get("platform_tag")
    if legacy_tag is not None and str(legacy_tag) not in out:
        legacy_entry = {
            key: payload.get(key)
            for key in (
                "source_version",
                "source",
                "source_stored",
                "shared_library",
                "platform",
                "machine",
                "platform_tag",
                "compiler",
                "command",
                "returncode",
                "stdout",
                "stderr",
                "status",
                "n_x",
                "n_y",
            )
            if key in payload
        }
        legacy_entry.setdefault("source_version", payload.get("source_version", _SOURCE_VERSION))
        legacy_entry.setdefault("platform_tag", str(legacy_tag))
        out[str(legacy_tag)] = legacy_entry
    return out


def _resolve_entry_shared_library(entry: dict[str, Any] | None, run_dir: Path) -> Path | None:
    if entry is None:
        return None
    if entry.get("status") != "ok":
        return None
    if entry.get("source_version") != _SOURCE_VERSION:
        return None
    shared_library = entry.get("shared_library")
    if not shared_library:
        return None

    candidate = Path(str(shared_library)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    run_local = run_dir / candidate
    if run_local.exists():
        return run_local

    legacy_relative = run_dir / str(shared_library)
    if legacy_relative.exists():
        return legacy_relative

    if entry.get("platform_tag") == _platform_tag():
        legacy_cached = _native_cache_dir() / candidate.name
        if legacy_cached.exists():
            return legacy_cached
    return None


def _resolve_shared_library(payload: dict[str, Any] | None, run_dir: Path) -> Path | None:
    current = _artifact_entries(payload).get(_platform_tag())
    return _resolve_entry_shared_library(current, run_dir)


def _is_usable_manifest(payload: dict[str, Any] | None, run_dir: Path) -> bool:
    return _resolve_shared_library(payload, run_dir) is not None


def _recompile_message(payload: dict[str, Any] | None, run_dir: Path) -> str | None:
    if payload is None:
        return None
    artifacts = _artifact_entries(payload)
    current = artifacts.get(_platform_tag())
    if current is not None:
        if current.get("source_version") != _SOURCE_VERSION:
            return "Native projection artifact is out of date. Compiling for current system."
        if _resolve_entry_shared_library(current, run_dir) is None:
            return "Native projection artifact for the current system was not found. Compiling for current system."
        return None
    if artifacts:
        return "Model was trained on a different system. Compiling for current system."
    return "Native projection artifact was not found. Compiling for current system."


def _relative_run_path(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def compile_native_projection(
    run_dir: str | Path,
    *,
    problem: dict[str, Any],
    settings: dict[str, Any],
    cc: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compile and store the native projection library in a platform-specific run artifact dir."""
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "projection_native.json"
    existing = _read_manifest(manifest_path)
    if not force and _is_usable_manifest(existing, target):
        return existing

    artifact_dir = _native_artifact_dir(target)
    n_x = len(_parameter_names(problem))
    n_y = len(problem.get("variables", []))
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        current = _platform_payload(status="cache-unavailable", stderr=f"{type(exc).__name__}: {exc}", n_x=n_x, n_y=n_y)
        payload = _manifest_payload(
            current=current,
            artifacts=_artifact_entries(existing),
            message=_recompile_message(existing, target),
        )
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    compilers = _compiler_candidates(cc)
    if not compilers:
        current = _platform_payload(status="missing-compiler", n_x=n_x, n_y=n_y)
        payload = _manifest_payload(
            current=current,
            artifacts=_artifact_entries(existing),
            message=_recompile_message(existing, target),
        )
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    shared_path = _native_library_path(target)
    message = _recompile_message(existing, target)
    artifacts = _artifact_entries(existing)
    last_payload: dict[str, Any] | None = None
    try:
        source_text = _source(problem, settings)
    except Exception as exc:
        current = _platform_payload(status="source-failed", stderr=f"{type(exc).__name__}: {exc}", n_x=n_x, n_y=n_y)
        payload = _manifest_payload(current=current, artifacts=artifacts, message=message)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    with tempfile.TemporaryDirectory(prefix="kkthn_native_") as tmp:
        source_path = Path(tmp) / "projection_native.c"
        source_path.write_text(source_text, encoding="utf-8")
        for compiler in compilers:
            if shared_path.exists():
                try:
                    shared_path.unlink()
                except OSError:
                    pass
            cmd = _compile_command(compiler, source_path, shared_path)
            try:
                proc = subprocess.run(cmd, cwd=str(target), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except OSError as exc:
                last_payload = _platform_payload(
                    status="compile-failed",
                    compiler=compiler,
                    command=cmd,
                    returncode=None,
                    stdout=None,
                    stderr=f"{type(exc).__name__}: {exc}",
                    n_x=n_x,
                    n_y=n_y,
                )
                continue
            last_payload = _platform_payload(
                shared_library=_relative_run_path(shared_path, target) if proc.returncode == 0 and shared_path.exists() else None,
                status="ok" if proc.returncode == 0 and shared_path.exists() else "compile-failed",
                compiler=compiler,
                command=cmd,
                returncode=int(proc.returncode),
                stdout=proc.stdout,
                stderr=proc.stderr,
                n_x=n_x,
                n_y=n_y,
            )
            if last_payload["status"] == "ok":
                break

    current = last_payload if last_payload is not None else _platform_payload(status="compile-failed", n_x=n_x, n_y=n_y)
    payload = _manifest_payload(current=current, artifacts=artifacts, message=message)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_native_projection(run_dir: str | Path, manifest: dict[str, Any] | None = None) -> NativeProjection | None:
    target = Path(run_dir)
    if manifest is None:
        manifest = _read_manifest(target / "projection_native.json")
    shared = _resolve_shared_library(manifest, target)
    if shared is None:
        return None
    current = _artifact_entries(manifest).get(_platform_tag(), manifest or {})
    return NativeProjection(shared, n_x=int(current["n_x"]), n_y=int(current["n_y"]))


def load_or_compile_native_projection(
    run_dir: str | Path,
    *,
    problem: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[NativeProjection | None, dict[str, Any]]:
    """Load a usable native projection library or try to compile one for this system."""
    target = Path(run_dir)
    manifest_path = target / "projection_native.json"
    manifest = _read_manifest(manifest_path)
    if not _is_usable_manifest(manifest, target):
        manifest = compile_native_projection(target, problem=problem, settings=settings)
    if _is_usable_manifest(manifest, target):
        try:
            projection = load_native_projection(target, manifest)
            if projection is None:
                raise FileNotFoundError("Native projection library is not available.")
            return projection, manifest
        except Exception as exc:
            current = _platform_payload(status="load-failed", stderr=f"{type(exc).__name__}: {exc}")
            failed = _manifest_payload(
                current=current,
                artifacts=_artifact_entries(manifest),
                message=manifest.get("message") if isinstance(manifest, dict) else None,
            )
            manifest_path.write_text(json.dumps(failed, indent=2, sort_keys=True), encoding="utf-8")
            return None, failed
    if manifest is not None:
        return None, manifest
    missing = _manifest_payload(current=_platform_payload(status="missing"))
    return None, missing
