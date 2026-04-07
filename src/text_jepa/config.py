from .tokenization import load_yaml_config


def get_model_settings(config_path):
    config = load_yaml_config(config_path)
    model_config = config.get("model") or {}

    hidden_dim = model_config.get("hidden_dim")
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("model.hidden_dim must be a positive integer")

    num_heads = model_config.get("num_heads")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("model.num_heads must be a positive integer")

    num_layers = model_config.get("num_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("model.num_layers must be a positive integer")

    ffn_dim = model_config.get("ffn_dim")
    if not isinstance(ffn_dim, int) or ffn_dim <= 0:
        raise ValueError("model.ffn_dim must be a positive integer")

    dropout = model_config.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in the range [0.0, 1.0)")

    norm = model_config.get("norm", "rms")
    if norm not in {"rms"}:
        raise ValueError("model.norm must be 'rms'")

    ema_momentum = model_config.get("ema_momentum", 0.996)
    if not isinstance(ema_momentum, (int, float)) or not 0.0 <= ema_momentum <= 1.0:
        raise ValueError("model.ema_momentum must be in the range [0.0, 1.0]")

    return {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "dropout": float(dropout),
        "norm": norm,
        "ema_momentum": float(ema_momentum),
    }
