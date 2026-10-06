"""Regression tests for the edge cases and crashes found during the error-free pass.

Each test below corresponds to a defect that was actually observed:
- degenerate light curves raised RuntimeWarnings / produced non-finite features
- robustness_score() returned NaN for an empty degradation list
- truncate_observation() raised IndexError when fraction > 1
- uncertainty_metrics() divided by zero for a single-class probability vector
- analyze_signal_quality() crashed on empty / list input
- report artifacts were written as cp1252 and broke downstream UTF-8 readers
- the dashboard showed a robustness metric that contradicted its own documented formula
- the analyst trace silently failed when the process CWD was not the project root
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from src.classical.feature_engineering import extract_all_features
from src.evaluation.degradation import truncate_observation, add_missing, add_gaussian_noise
from src.evaluation.robustness_score import robustness_score
from src.evaluation.uncertainty import uncertainty_metrics
from src.fusion.signal_quality import analyze_signal_quality

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


DEGENERATE_CURVES = {
    "constant": np.full(256, 0.5),
    "zeros": np.zeros(256),
    "ones": np.ones(256),
    "empty": np.array([]),
    "single": np.array([0.5]),
    "two": np.array([0.1, 0.9]),
    "all_nan": np.full(256, np.nan),
    "inf": np.concatenate([np.full(64, np.inf), np.linspace(0, 1, 192)]),
    "mixed_nan": np.concatenate([np.full(128, np.nan), np.linspace(0, 1, 128)]),
    "negative": np.linspace(-1.0, 1.0, 256),
}


@pytest.mark.parametrize("name", sorted(DEGENERATE_CURVES))
def test_features_finite_on_degenerate_input(name):
    """Feature extraction must always return a finite 21-dim vector, never raise or emit warnings."""
    f = np.asarray(extract_all_features(DEGENERATE_CURVES[name], sampling_interval=720 / 255), dtype=float)
    assert f.shape == (21,)
    assert np.isfinite(f).all(), f"non-finite features for {name}: {f}"


def test_features_no_runtime_warning_on_constant_curve():
    """A constant signal has undefined moments; we must not emit a scipy precision-loss warning."""
    with np.errstate(all="raise"):
        f = extract_all_features(np.full(256, 0.5), sampling_interval=720 / 255)
    assert np.isfinite(np.asarray(f, dtype=float)).all()


def test_robustness_score_empty_degradation_list():
    """No degradation conditions -> 0.0, not NaN."""
    score = robustness_score(0.85, [])
    assert np.isfinite(score)
    assert score == 0.0


def test_robustness_score_zero_clean_accuracy():
    assert robustness_score(0.0, [0.5, 0.6]) == 0.0


def test_robustness_score_normal_range():
    score = robustness_score(0.80, [0.72, 0.64])
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx((0.72 / 0.80 + 0.64 / 0.80) / 2)


def test_truncate_observation_clamps_out_of_range_fraction():
    curves = np.random.RandomState(0).rand(4, 256)
    for frac in (0.0, -0.5, 0.25, 1.0, 1.5, 3.0):
        out = truncate_observation(curves, frac)
        assert out.shape == curves.shape, f"fraction {frac} broke the shape"


def test_degradation_helpers_preserve_shape():
    curves = np.random.RandomState(0).rand(6, 256)
    assert add_gaussian_noise(curves, 0.0).shape == curves.shape
    assert add_missing(curves, 0.0).shape == curves.shape
    assert add_missing(curves, 0.2).shape == curves.shape


def test_uncertainty_metrics_single_class():
    m = uncertainty_metrics(np.array([1.0]))
    assert np.isfinite(m["norm_entropy"])
    assert 0.0 <= m["norm_entropy"] <= 1.0
    assert m["uncertainty_label"] in ("Low", "Moderate", "High")


def test_uncertainty_metrics_empty_input():
    m = uncertainty_metrics(np.array([]))
    assert m["uncertainty_label"] == "High"
    assert np.isfinite(m["norm_entropy"])


def test_uncertainty_metrics_uniform_is_high_uncertainty():
    m = uncertainty_metrics(np.full(5, 0.2))
    assert m["uncertainty_label"] == "High"
    assert m["predicted_class"] == 0


def test_signal_quality_empty_curve():
    sq = analyze_signal_quality(np.array([]))
    assert sq["quality_label"] == "Low"
    assert 0.0 <= sq["signal_quality_score"] <= 1.0


def test_signal_quality_accepts_list_input():
    sq = analyze_signal_quality([0.1, 0.9, 0.4, 0.7])
    assert sq["quality_label"] in ("Low", "Medium", "High")


def test_signal_quality_counts_missing_samples():
    curve = np.concatenate([np.full(64, np.nan), np.linspace(0, 1, 192)])
    sq = analyze_signal_quality(curve)
    assert sq["missing_pct"] == pytest.approx(64 / 256)


def test_signal_quality_score_bounds():
    sq = analyze_signal_quality(np.random.RandomState(1).rand(256))
    assert 0.0 <= sq["signal_quality_score"] <= 1.0
    assert np.isfinite(sq["noise_estimate"])


# ---------------------------------------------------------------------------
# Encoding: every text artifact must be valid UTF-8. Six report files were once
# written as cp1252 (em-dash = 0x97) and crashed the UTF-8 report generator.
# ---------------------------------------------------------------------------

TEXT_SUFFIXES = {".md", ".json", ".csv", ".txt"}


def test_report_artifacts_are_valid_utf8():
    bad = []
    for root in ("results", "reports"):
        for p in (PROJECT_ROOT / root).rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if "__pycache__" in str(p):
                continue
            try:
                p.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                bad.append(str(p.relative_to(PROJECT_ROOT)))
    assert not bad, f"non-UTF-8 text artifacts: {bad}"


def test_report_writers_declare_utf8_encoding():
    """Text-mode writes must pin encoding so Windows' cp1252 default can't corrupt artifacts."""
    offenders = []
    for py in (PROJECT_ROOT / "experiments").glob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if "open(" in line and '"w"' in line and "encoding=" not in line:
                offenders.append(f"{py.name}:{i}")
    assert not offenders, f"text writes without encoding=: {offenders}"


# ---------------------------------------------------------------------------
# Dashboard metric integrity: the robustness value shown on the comparison
# cards must come from the documented formula, not a second local variant.
# ---------------------------------------------------------------------------


def test_dashboard_uses_shared_robustness_formula():
    src = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    assert "from src.evaluation.robustness_score import robustness_score" in src, (
        "dashboard must reuse src/evaluation/robustness_score.py as the single definition of RS"
    )


def test_documented_robustness_formula_is_mean_retention():
    """RS = mean(acc_degraded / acc_clean) — pin the documented behaviour."""
    assert robustness_score(1.0, [0.5, 0.5]) == pytest.approx(0.5)
    assert robustness_score(0.8, [0.8, 0.8, 0.8]) == pytest.approx(1.0)
    # using only the worst condition would give 0.5 here, so this pins the mean
    assert robustness_score(1.0, [0.9, 0.5]) == pytest.approx(0.7)


def test_robustness_sweeps_share_clean_baseline():
    """Noise / observation / missing sweeps must agree on the clean row, so aggregating is sound."""
    reports = PROJECT_ROOT / "results" / "reports"
    baselines = {}
    for fn in ("robustness_noise.json", "robustness_observation.json", "robustness_missing.json"):
        p = reports / fn
        if not p.exists():
            pytest.skip(f"{fn} not generated yet")
        data = json.loads(p.read_text(encoding="utf-8"))
        baselines[fn] = {k: v for k, v in data[0].items() if k in ("Classical_CNN", "Pure_Quantum", "Hybrid")}
    vals = list(baselines.values())
    for other in vals[1:]:
        assert other == vals[0], f"clean baselines differ across sweeps: {baselines}"


# ---------------------------------------------------------------------------
# AI Analyst traceability: the trace must land in the project root no matter
# what the process CWD is (it previously used a CWD-relative path and failed
# silently, breaking the traceability claim).
# ---------------------------------------------------------------------------


def test_analyst_trace_is_written_regardless_of_cwd(tmp_path, monkeypatch):
    from src.agents.analyst import analyze

    monkeypatch.chdir(tmp_path)
    text = analyze({
        "classical_pred": 2, "hybrid_pred": 2, "quantum_pred": 2,
        "uncertainty": {"uncertainty_label": "Low", "entropy": 0.2,
                        "margin": 0.8, "probability": 0.9},
        "agreement": {"consensus": "3/3"},
        "signal_quality": "High",
    })
    assert text
    trace = PROJECT_ROOT / "results" / "reports" / "analyst_trace.json"
    assert trace.exists(), "analyst trace was not written to the project root"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert "input_trace" in payload and "output_lines" in payload


def test_backends_package_does_not_eagerly_import_aer():
    """Importing src.quantum.backends must not pull in qiskit_aer.

    qiskit_aer calls aer_initialize_libraries() at import time. With conflicting
    OpenMP runtimes present (vcomp140.dll from sklearn/faiss vs libiomp5md.dll from
    torch) that native call hard-crashes the interpreter, and `except Exception`
    cannot catch it. Keeping the Aer import lazy confines the blast radius to code
    that explicitly asks for Aer.
    """
    code = (
        "import sys; sys.path.insert(0, r'{root}');"
        "import src.quantum.backends as b;"
        "b.get_backend('pennylane'); b.get_backend('stim');"
        "print('AER_LOADED' if any(m.split('.')[0] == 'qiskit_aer' for m in sys.modules)"
        " else 'AER_NOT_LOADED')"
    ).format(root=PROJECT_ROOT)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"import crashed: rc={out.returncode} {out.stderr[-400:]}"
    assert "AER_NOT_LOADED" in out.stdout, f"qiskit_aer was imported eagerly: {out.stdout}"


def test_backend_lookup_by_name():
    import src.quantum.backends as backends

    assert backends.get_backend("pennylane").__name__ == "PennylaneBackend"
    assert backends.get_backend("stim").__name__ == "StimBackend"
    assert backends.get_backend("PENNYLANE").__name__ == "PennylaneBackend"
    with pytest.raises(ValueError):
        backends.get_backend("does-not-exist")


def test_lab_page_has_no_eager_aer_import():
    """Quantum Lab / playground pages were removed from the active site.

    They must not exist under app/pages/ (archived copies under app/_archive/
    are fine — they never execute). Remaining pages must not import the Aer
    backend at module level (unused + crash risk).
    """
    lab_pages = list((PROJECT_ROOT / "app" / "pages").glob("*Lab*.py"))
    assert not lab_pages, f"lab pages still active: {lab_pages}"
    for p in (PROJECT_ROOT / "app" / "pages").glob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert "from src.quantum.backends.aer_backend import AerBackend" not in src, (
            f"{p.name} imports AerBackend at module level"
        )


def test_no_bare_except_in_active_code():
    """A bare except clause also swallows KeyboardInterrupt/SystemExit; none should remain."""
    import re
    pat = re.compile(r"\bexcept\s*:")
    offenders = []
    this_file = pathlib.Path(__file__).resolve()
    for folder in ("app", "src", "experiments", "reports", "tests"):
        for py in (PROJECT_ROOT / folder).rglob("*.py"):
            if "__pycache__" in str(py) or py.name in ("app_backup.py", "quantum_lab_full.py"):
                continue
            if py.resolve() == this_file:
                continue  # this test necessarily mentions the pattern
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(line):
                    offenders.append(f"{py.relative_to(PROJECT_ROOT)}:{i}")
    assert not offenders, f"bare except clauses: {offenders}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
