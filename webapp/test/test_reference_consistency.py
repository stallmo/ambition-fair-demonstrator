"""Cross-checks that the Qt-free ported modules are numerically consistent
with the original desktop app's PyQt5-based ``core/`` modules, given the
same fixed RNG seeds and inputs.

The webapp's own virtual environment deliberately has no PyQt5
dependency (that's the whole point of story A2), so these tests shell
out to a *separate* process using the original desktop project's own
virtual environment (which does have PyQt5) to run the reference
implementation, capture its output as JSON, and compare it against the
ported implementation run in-process here. If that reference
environment isn't available (e.g. a minimal CI image without the
desktop app's venv set up), the tests are skipped rather than failed --
the Qt-free-ness of the port is independently covered by
``test_no_pyqt5_imports.py``.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# webapp/test/ -> webapp/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROOT_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"


def _root_python_available() -> bool:
    """Check whether the desktop app's own venv (with PyQt5) is usable for cross-checks.

    :returns: True if the root project's venv exists and can import PyQt5.
    :rtype: bool
    """
    if not _ROOT_PYTHON.exists():
        return False
    try:
        result = subprocess.run(
            [str(_ROOT_PYTHON), "-c", "import PyQt5"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _root_python_available(),
    reason="Original desktop app's PyQt5 virtual environment is not available for cross-checking",
)


def _run_reference_script(script: str) -> Any:
    """Run ``script`` with the desktop app's own python and parse its JSON stdout.

    :param script: Python source to execute via ``python -c``.
    :type script: str
    :returns: The JSON-decoded stdout of the script.
    :rtype: Any
    """
    result = subprocess.run(
        [str(_ROOT_PYTHON), "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"reference script failed:\n{result.stderr}"
    return json.loads(result.stdout)


def test_data_generator_tick_matches_original_generate_tick() -> None:
    """DataGenerator.tick() must reproduce the original _generate_tick() output exactly."""
    reference_script = """
import json
import numpy as np
from core.data_generator import DataGenerator, StreamConfig

gen = DataGenerator()
gen._rng = np.random.default_rng(42)
gen.configure(StreamConfig(mean=70.0, std=2.0, noise_amplitude=0.5,
                            anomaly_probability=0.2, anomaly_magnitude=15.0, unit="C"))
results = []
def capture(ts, value, is_injected):
    results.append([value, bool(is_injected)])
gen.new_sample.connect(capture)
for _ in range(200):
    gen._generate_tick()
print(json.dumps(results))
"""
    reference = _run_reference_script(reference_script)

    from core.data_generator import DataGenerator, StreamConfig

    gen = DataGenerator(rng=np.random.default_rng(42))
    gen.configure(
        StreamConfig(
            mean=70.0, std=2.0, noise_amplitude=0.5,
            anomaly_probability=0.2, anomaly_magnitude=15.0, unit="C",
        )
    )
    ported = [list(gen.tick()[1:]) for _ in range(200)]

    assert len(ported) == len(reference) == 200
    injected_count = 0
    for (p_value, p_injected), (r_value, r_injected) in zip(ported, reference):
        assert p_injected == r_injected
        assert p_value == pytest.approx(r_value, abs=1e-9)
        injected_count += int(p_injected)
    # Sanity: with probability 0.2 over 200 samples, the injection branch
    # (and thus the rng.choice() call) must have actually been exercised.
    assert injected_count > 0


def test_anomaly_detector_feed_and_params_match_original() -> None:
    """AnomalyDetector.feed()/get_model_params() must match the original implementation."""
    reference_script = """
import json
import numpy as np
from core.anomaly_detector import AnomalyDetector

det = AnomalyDetector()
results = []
def capture(res):
    results.append([res.score, res.confidence, res.is_anomaly])
det.detection_ready.connect(capture)

rng = np.random.default_rng(123)
vals = rng.normal(70, 2, size=130)
for i, v in enumerate(vals):
    det.feed(float(i), float(v), False)

params = det.get_model_params()
print(json.dumps({"results": results, "params": params}))
"""
    reference = _run_reference_script(reference_script)

    from core.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    rng = np.random.default_rng(123)
    vals = rng.normal(70, 2, size=130)
    ported_results = []
    for i, v in enumerate(vals):
        result = det.feed(float(i), float(v), False)
        ported_results.append([result.score, result.confidence, result.is_anomaly])
    ported_params = det.get_model_params()

    assert len(ported_results) == len(reference["results"]) == 130
    for (p_score, p_conf, p_anom), (r_score, r_conf, r_anom) in zip(
        ported_results, reference["results"]
    ):
        assert p_anom == r_anom
        assert p_score == pytest.approx(r_score, abs=1e-9)
        assert p_conf == pytest.approx(r_conf, abs=1e-9)

    assert ported_params is not None
    ref_params = reference["params"]
    assert ref_params is not None
    assert ported_params["means"] == pytest.approx(ref_params["means"], abs=1e-9)
    assert ported_params["stds"] == pytest.approx(ref_params["stds"], abs=1e-9)
    assert ported_params["weights"] == pytest.approx(ref_params["weights"], abs=1e-9)
    assert ported_params["boundary_values"] == pytest.approx(ref_params["boundary_values"], abs=1e-6)


def test_anomaly_detector_retrain_matches_original() -> None:
    """AnomalyDetector.retrain(start, end) must match the original's retrain output."""
    reference_script = """
import json
import numpy as np
from core.anomaly_detector import AnomalyDetector

det = AnomalyDetector()
rng = np.random.default_rng(9)
vals = rng.normal(100, 3, size=150)
for i, v in enumerate(vals):
    det.feed(float(i), float(v), False)
det.retrain(10, 80)
print(json.dumps(det.get_model_params()))
"""
    reference = _run_reference_script(reference_script)

    from core.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    rng = np.random.default_rng(9)
    vals = rng.normal(100, 3, size=150)
    for i, v in enumerate(vals):
        det.feed(float(i), float(v), False)
    det.retrain(10, 80)
    ported_params = det.get_model_params()

    assert ported_params["means"] == pytest.approx(reference["means"], abs=1e-9)
    assert ported_params["stds"] == pytest.approx(reference["stds"], abs=1e-9)
    assert ported_params["weights"] == pytest.approx(reference["weights"], abs=1e-9)
