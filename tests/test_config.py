from text_jepa.config import get_model_settings

from conftest import write_test_config


def test_get_model_settings_returns_expected_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_test_config(
        config_path,
        hidden_dim=16,
        num_heads=4,
        num_layers=3,
        ffn_dim=64,
        dropout=0.1,
        ema_momentum=0.99,
    )

    settings = get_model_settings(config_path)

    assert settings == {
        "hidden_dim": 16,
        "num_heads": 4,
        "num_layers": 3,
        "ffn_dim": 64,
        "dropout": 0.1,
        "norm": "rms",
        "ema_momentum": 0.99,
    }


def test_get_model_settings_rejects_invalid_ema_momentum(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_test_config(config_path, ema_momentum=1.5)

    try:
        get_model_settings(config_path)
    except ValueError as exc:
        assert "model.ema_momentum" in str(exc)
    else:
        raise AssertionError("Expected get_model_settings to reject an invalid ema_momentum")
