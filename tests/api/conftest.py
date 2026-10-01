"""Stand-ins the serving application's tests are built from.

What these tests are about is where a model comes from and how long the
service built from it lives, not what the model predicts. So the artifact
is never real: loading is replaced by a recorder that reports which path it
was asked for and how many times, and what it returns is a stand-in with
just enough of a fitted pipeline's surface for a ``PredictionService`` to
use it.

That replacement is also what keeps the suite independent of a trained
model being present on disk, which no test should have to arrange and no
checkout is guaranteed to contain.
"""

from pathlib import Path

import numpy as np
import pytest

from churner.api import app as app_module
from churner.config.settings import Settings

# --- The configuration the tests run an application under ---
# A path no checkout contains, so a test that reached the real filesystem
# would fail rather than quietly load something.
MODEL_PATH = Path("artifacts/stand-in-churn-model.joblib")
MODEL_VERSION = "churn-stand-in-2026-10-01"

CHURN_PREDICTION = "Yes"
CHURN_PROBABILITY = 0.73


class StubPipeline:
    """A stand-in for a fitted pipeline, returning predetermined figures.

    Only what ``PredictionService`` reads is present: the class ordering,
    predicted labels, and probability columns laid out in that order.
    """

    def __init__(self) -> None:
        self.classes_ = np.array(["No", "Yes"])

    def predict(self, features):
        return np.array([CHURN_PREDICTION] * len(features))

    def predict_proba(self, features):
        return np.tile([1.0 - CHURN_PROBABILITY, CHURN_PROBABILITY], (len(features), 1))


class RecordingLoader:
    """A stand-in for ``load_model`` that records what it was asked for.

    Every path it is called with is kept, so a test can state not only that
    a model was loaded but that it was the configured one and that it was
    loaded exactly once.
    """

    def __init__(self, model=None, error: Exception | None = None) -> None:
        self.requested_paths: list[Path] = []
        self._model = model
        self._error = error

    def __call__(self, path):
        self.requested_paths.append(path)
        if self._error is not None:
            raise self._error
        return self._model

    @property
    def call_count(self) -> int:
        """How many times a model was asked for."""
        return len(self.requested_paths)


@pytest.fixture
def settings() -> Settings:
    """Configure an application with a path and version of the test's own."""
    return Settings(model_path=MODEL_PATH, model_version=MODEL_VERSION)


@pytest.fixture
def loader(monkeypatch) -> RecordingLoader:
    """Replace the application's loader with one that returns a stand-in."""
    recorder = RecordingLoader(StubPipeline())
    monkeypatch.setattr(app_module, "load_model", recorder)
    return recorder


@pytest.fixture
def failing_loader(monkeypatch) -> RecordingLoader:
    """Replace the loader with one that fails the way a missing artifact does."""
    recorder = RecordingLoader(
        error=FileNotFoundError(f"No model artifact to load at {MODEL_PATH}")
    )
    monkeypatch.setattr(app_module, "load_model", recorder)
    return recorder
