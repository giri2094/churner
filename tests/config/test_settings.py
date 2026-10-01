"""Unit tests for the configuration the serving application runs under.

These tests establish the public contract of ``Settings`` and
``load_settings``: both values are always available, they come from the
environment when it states them, and they fall back to defaults that belong
to no particular machine when it does not.

Environments are supplied as plain mappings rather than installed into the
process, which is what makes a stated environment visible in the test that
depends on it. One case sets real variables and passes no mapping, because
reading the process environment by default is itself part of the contract.

Nothing here loads a model, builds an application, or checks that a
configured path exists: configuration reports what it was told, and whether
an artifact is there is answered when one is loaded.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from churner.config.settings import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_VERSION,
    MODEL_PATH_VARIABLE,
    MODEL_VERSION_VARIABLE,
    Settings,
    load_settings,
)

# --- An environment that configures both values ---
# Neither value resembles a default, so a returned one can only have been
# read rather than fallen back to.
CONFIGURED_MODEL_PATH = "artifacts/churn-model-2026-10-01.joblib"
CONFIGURED_MODEL_VERSION = "churn-logistic-2026-10-01"

CONFIGURED_ENVIRONMENT = {
    MODEL_PATH_VARIABLE: CONFIGURED_MODEL_PATH,
    MODEL_VERSION_VARIABLE: CONFIGURED_MODEL_VERSION,
}

EMPTY_ENVIRONMENT: dict[str, str] = {}


# --- Both values are always available ---


def test_settings_hold_a_model_path_and_a_model_version():
    """The object states the two values the application is built from."""
    settings = Settings(model_path=Path("models/churn_model.joblib"), model_version="v1")

    assert settings.model_path == Path("models/churn_model.joblib")
    assert settings.model_version == "v1"


def test_loaded_settings_hold_both_values_when_nothing_is_configured():
    """An unconfigured environment still yields a usable pair."""
    settings = load_settings(EMPTY_ENVIRONMENT)

    assert isinstance(settings.model_path, Path)
    assert isinstance(settings.model_version, str)
    assert settings.model_version != ""


# --- What the environment says is what is used ---


def test_configured_values_are_read_from_the_environment():
    """Both variables are taken from the environment that states them."""
    settings = load_settings(CONFIGURED_ENVIRONMENT)

    assert settings.model_path == Path(CONFIGURED_MODEL_PATH)
    assert settings.model_version == CONFIGURED_MODEL_VERSION


def test_the_model_path_arrives_as_a_path_rather_than_the_string_read():
    """``load_model`` is handed a ``Path``, so that is what is configured."""
    assert isinstance(load_settings(CONFIGURED_ENVIRONMENT).model_path, Path)


def test_the_process_environment_is_read_when_no_mapping_is_given(monkeypatch):
    """Called with no argument, the process environment is the source."""
    monkeypatch.setenv(MODEL_PATH_VARIABLE, CONFIGURED_MODEL_PATH)
    monkeypatch.setenv(MODEL_VERSION_VARIABLE, CONFIGURED_MODEL_VERSION)

    settings = load_settings()

    assert settings.model_path == Path(CONFIGURED_MODEL_PATH)
    assert settings.model_version == CONFIGURED_MODEL_VERSION


# --- Defaults, and what they are not ---


def test_defaults_are_used_when_the_environment_states_nothing():
    """An empty environment gives the stated defaults, not something else."""
    settings = load_settings(EMPTY_ENVIRONMENT)

    assert settings.model_path == DEFAULT_MODEL_PATH
    assert settings.model_version == DEFAULT_MODEL_VERSION


def test_a_blank_variable_is_treated_as_unset():
    """A variable defined to nothing configures nothing.

    Without this, an environment exporting an empty value would configure a
    path of ``""``, which resolves to the working directory and fails later
    as a confusing read rather than immediately as a default.
    """
    blank_environment = {MODEL_PATH_VARIABLE: "   ", MODEL_VERSION_VARIABLE: ""}

    settings = load_settings(blank_environment)

    assert settings.model_path == DEFAULT_MODEL_PATH
    assert settings.model_version == DEFAULT_MODEL_VERSION


def test_the_default_model_path_is_not_an_absolute_filesystem_path():
    """The default resolves against a deployment's directory, not a machine's.

    An absolute default would name one developer's disk and be wrong in
    every other environment while still looking configured.
    """
    assert not load_settings(EMPTY_ENVIRONMENT).model_path.is_absolute()


# --- Configuration does not change under a running application ---


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [("model_path", Path("elsewhere.joblib")), ("model_version", "churn-other")],
)
def test_settings_refuse_field_assignment(field_name, new_value):
    """Both fields are frozen, so a running application cannot be re-pointed."""
    settings = load_settings(CONFIGURED_ENVIRONMENT)

    with pytest.raises(FrozenInstanceError):
        setattr(settings, field_name, new_value)

    assert getattr(settings, field_name) != new_value
