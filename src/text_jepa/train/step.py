from ..utils.ema import update_ema


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    update_ema(model.target_tower, model.context_tower, model.ema_momentum)

    return outputs
