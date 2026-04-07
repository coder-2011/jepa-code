def train_llm_jepa_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    outputs = model(**batch)
    outputs["loss"].backward()
    optimizer.step()

    return outputs
