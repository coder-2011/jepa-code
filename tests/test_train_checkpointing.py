from pathlib import Path
import importlib.util

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "train.py"
SPEC = importlib.util.spec_from_file_location("train_script", TRAIN_PATH)
train_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_script)


def test_checkpoint_roundtrip_restores_model_optimizer_and_step(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)

    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

    run_config = {"steps": 10}
    train_script.save_checkpoint(
        tmp_path,
        3,
        train_script.checkpoint_state(3, model, optimizer, run_config),
    )

    restored_model = torch.nn.Linear(4, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_step = train_script.load_checkpoint(
        tmp_path / "latest.pt",
        restored_model,
        restored_optimizer,
        "cpu",
    )

    assert restored_step == 3
    assert (tmp_path / "step-000003.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    for before, after in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(before, after)
    assert restored_optimizer.state_dict()["state"]


def test_resolve_resume_path_prefers_explicit_path_and_supports_auto_resume(tmp_path):
    explicit = tmp_path / "explicit.pt"
    explicit.write_text("x", encoding="utf-8")
    latest = tmp_path / "latest.pt"
    latest.write_text("y", encoding="utf-8")

    assert train_script.resolve_resume_path(explicit, tmp_path, auto_resume=True) == explicit
    assert train_script.resolve_resume_path(None, tmp_path, auto_resume=True) == latest
    assert train_script.resolve_resume_path(None, tmp_path, auto_resume=False) is None


def test_evaluate_averages_loss_and_restores_train_mode():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._losses = iter([torch.tensor(2.0), torch.tensor(4.0)])

        def forward(self, **_batch):
            return {"loss": next(self._losses)}

    dataloader = [
        {"x": torch.tensor([1.0])},
        {"x": torch.tensor([2.0])},
    ]
    model = DummyModel()
    model.train()

    loss = train_script.evaluate(model, dataloader, "cpu", max_batches=2)

    assert loss == 3.0
    assert model.training
