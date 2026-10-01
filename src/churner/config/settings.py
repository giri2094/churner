"""The deployment-specific values the serving application is given.

This module is the one place that reads the environment. Which artifact a
deployment serves and what it calls that artifact are properties of the
deployment rather than of the code, so they arrive from outside; gathering
them here means the rest of the application is handed a value it can read
instead of a variable name it has to know.

The values are returned as one frozen object rather than fetched where they
are needed. Scattered ``os.getenv`` calls would spread the variable names
through the modules that happen to use them, let two of them disagree about
a default, and leave a test with no way to supply a value except by editing
the process environment.

Defaults are relative paths and neutral strings, never a path from one
machine. An absolute default would work in exactly one place and silently
point somewhere wrong everywhere else, which is worse than not resolving at
all; a relative default resolves against the working directory a deployment
chooses, and a missing artifact there is reported when it is loaded.

Nothing here loads a model, validates that the configured artifact exists,
or decides when the configuration is read. Reading a value is not the same
as acting on it, and acting on it belongs to the startup that does.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ
from pathlib import Path

# --- The variables a deployment sets ---
MODEL_PATH_VARIABLE = "CHURNER_MODEL_PATH"
MODEL_VERSION_VARIABLE = "CHURNER_MODEL_VERSION"

# --- What is used when a deployment sets nothing ---
# The path is relative, so it resolves against the directory the application
# is run from rather than against one machine's filesystem. The version says
# that the deployment did not name one, which is the honest stamp to put on
# a result: a plausible-looking default would misattribute it.
DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")
DEFAULT_MODEL_VERSION = "unversioned"


@dataclass(frozen=True)
class Settings:
    """The configuration the serving application runs under.

    ``model_path`` names the artifact to load, filename included, in the
    form ``load_model`` reads. ``model_version`` identifies the model that
    artifact holds and is what every prediction is stamped with; it is a
    deployment's statement about the artifact, not something read from it.

    The object is frozen because configuration is settled once, before the
    application starts: a value that changed underneath a running service
    would leave results stamped with a version the model never had.
    """

    model_path: Path
    model_version: str


def load_settings(environment: Mapping[str, str] | None = None) -> Settings:
    """Read the configuration from environment variables.

    A variable that is unset, empty, or blank is treated as not given and
    falls back to its default, so an environment that defines a variable to
    nothing behaves as one that never defined it rather than configuring a
    path of ``""``.

    Parameters
    ----------
    environment : Mapping[str, str] | None
        Where to read the variables from. Defaults to the process
        environment; a caller passes a mapping to state an environment
        rather than to install one. Only read from.

    Returns
    -------
    Settings
        The configured artifact path and model version.
    """
    source = environ if environment is None else environment

    return Settings(
        model_path=Path(_value(source, MODEL_PATH_VARIABLE, str(DEFAULT_MODEL_PATH))),
        model_version=_value(source, MODEL_VERSION_VARIABLE, DEFAULT_MODEL_VERSION),
    )


def _value(environment: Mapping[str, str], variable: str, default: str) -> str:
    """Return a variable's value, or ``default`` if it carries none."""
    return environment.get(variable, "").strip() or default
