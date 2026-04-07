import torch
from torch import nn

from text_jepa.utils.ema import update_ema


class TinyModule(nn.Module):
    def __init__(self, weight_value, buffer_value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight_value], dtype=torch.float32))
        self.register_buffer("step", torch.tensor([buffer_value], dtype=torch.float32))


def test_update_ema_with_zero_momentum_copies_source():
    target = TinyModule(weight_value=0.0, buffer_value=0.0)
    source = TinyModule(weight_value=2.0, buffer_value=3.0)

    update_ema(target, source, momentum=0.0)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.step, source.step)


def test_update_ema_interpolates_parameters_and_copies_buffers():
    target = TinyModule(weight_value=0.0, buffer_value=0.0)
    source = TinyModule(weight_value=2.0, buffer_value=5.0)

    update_ema(target, source, momentum=0.5)

    assert torch.equal(target.weight, torch.tensor([1.0]))
    assert torch.equal(target.step, source.step)


def test_update_ema_rejects_invalid_momentum():
    target = TinyModule(weight_value=0.0, buffer_value=0.0)
    source = TinyModule(weight_value=2.0, buffer_value=3.0)

    try:
        update_ema(target, source, momentum=1.5)
    except ValueError as exc:
        assert "momentum" in str(exc)
    else:
        raise AssertionError("Expected update_ema to reject an invalid momentum")
