from text_jepa.utils.ema import update_ema


def train_llm_jepa_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    outputs = model(**batch)
    outputs["loss"].backward()
    optimizer.step()
    update_ema(model.target_backbone, model.backbone, model.ema_momentum)

    return outputs
