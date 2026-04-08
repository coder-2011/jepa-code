from text_jepa.utils.ema import update_ema


def train_llm_jepa_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)
    outputs["loss"].backward()
    optimizer.step()
    if getattr(model, "uses_target_backbone", False):
        update_ema(model.target_backbone, model.backbone, model.ema_momentum)

    return outputs
