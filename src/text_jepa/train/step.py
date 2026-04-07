from ..utils.ema import update_ema


def train_step(model, optimizer, batch):
    model.train()
    # Default zero_grad behavior is sufficient here because the training step owns the full backward pass.
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    # EMA must run after the optimizer so the teacher tracks the freshly updated student weights.
    update_ema(model.target_tower, model.context_tower, model.ema_momentum)

    return outputs
