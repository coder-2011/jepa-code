import torch

from text_jepa.models.layer_model import LayerModel
from text_jepa.train.step import train_step


def make_model():
    return LayerModel(
        vocab_size=32,
        max_length=16,
        hidden_dim=8,
        encoder_num_layers=2,
        encoder_num_heads=2,
        encoder_ffn_dim=32,
        predictor_num_layers=2,
        predictor_num_heads=2,
        predictor_ffn_dim=32,
        ema_momentum=0.9,
    )


def make_batch():
    return {
        "input_ids_full": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long),
        "input_ids_ctx": torch.tensor([[1, 9, 3, 4], [5, 6, 9, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long),
        "target_positions": torch.tensor([[1, 3], [2, 0]], dtype=torch.long),
        "target_valid_mask": torch.tensor([[True, False], [True, False]]),
    }


def test_train_step_returns_loss_and_updates_target_tower():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_batch()

    target_before = [parameter.detach().clone() for parameter in model.target_tower.parameters()]
    outputs = train_step(model, optimizer, batch)

    assert outputs["loss"].ndim == 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, model.target_tower.parameters())
    )
