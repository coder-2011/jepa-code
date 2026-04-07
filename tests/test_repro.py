import os

from text_jepa.utils.repro import configure_reproducibility, resolve_deterministic, resolve_seed


def test_resolve_seed_prefers_config_when_cli_override_missing():
    config = {"runtime": {"seed": 17}}

    assert resolve_seed(config, None) == 17


def test_resolve_seed_prefers_cli_override():
    config = {"runtime": {"seed": 17}}

    assert resolve_seed(config, 23) == 23


def test_resolve_deterministic_uses_runtime_default():
    config = {"runtime": {"deterministic": False}}

    assert resolve_deterministic(config, None) is False


def test_resolve_deterministic_prefers_cli_override():
    config = {"runtime": {"deterministic": False}}

    assert resolve_deterministic(config, True) is True


def test_configure_reproducibility_sets_python_hash_seed():
    configure_reproducibility(11, deterministic=True)

    assert os.environ["PYTHONHASHSEED"] == "11"
