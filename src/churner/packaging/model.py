"""Persistence of a fitted model pipeline to a file, and back from one.

This module is the step that sits after selection: it writes a fitted
pipeline to a path and reads one back. Training deliberately does not
persist what it fits, so this is the only place in the workflow where a
model outlives the process that produced it.

What is written is the whole fitted ``Pipeline`` -- the preprocessing it
learned together with the classifier fitted on what that preprocessing
produced -- rather than the final estimator alone. An estimator saved
without its preprocessing could only be handed an already-transformed
matrix, which would move the transformation into the caller and let it
drift from the one the fit was performed under. Saving the pipeline whole
is what keeps a loaded model able to predict from a raw feature frame.

The destination is an argument rather than a constant. Where an artifact
belongs is the caller's decision: a script writing into ``models/``, a test
writing into a temporary directory, a container reading a mounted volume. A
default stated here would be one of those choices imposed on the rest.

Serialisation is joblib's rather than ``pickle``'s, because a fitted
pipeline carries large NumPy arrays and joblib stores those far more
efficiently than the general-purpose pickler does. It is still a pickle
underneath, so an artifact is only readable by a compatible environment and
only as trustworthy as its source.

Nothing here trains a model, evaluates one, decides which model to save,
chooses where it goes, records what produced it, or serves it.
"""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, path: str | Path) -> None:
    """Write a fitted pipeline to ``path``.

    The pipeline is serialised whole, preprocessing included, so that what
    is read back predicts from a raw feature frame exactly as the object
    handed over does. Parent directories that do not exist are created, so
    a caller does not have to prepare the location first, and an artifact
    already at ``path`` is replaced.

    The model is only read from. Saving does not fit it, refit it, or
    otherwise change its state.

    Parameters
    ----------
    model : Pipeline
        The fitted pipeline to persist. Only read from.
    path : str | Path
        Where to write the artifact, filename included.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> Pipeline:
    """Read back a fitted pipeline that ``save_model`` wrote.

    The object returned is reconstructed from the file rather than being
    the one that was saved, and it arrives fitted: it predicts without
    being fitted again.

    A missing artifact is reported as a missing artifact. A wrong path is
    the ordinary way this fails, and the read error underneath would not
    say which file was expected or where it should have come from.

    Parameters
    ----------
    path : str | Path
        The artifact to read, filename included.

    Returns
    -------
    Pipeline
        The fitted pipeline stored in the file.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``path``.
    """
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"No model artifact to load at {source}. load_model reads a file "
            "written by save_model; it does not train or create one."
        )

    return joblib.load(source)
