import torch


@torch.no_grad()
def update_ema(target_module, source_module, momentum):
    if not isinstance(momentum, (int, float)) or not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in the range [0.0, 1.0]")

    target_parameters = dict(target_module.named_parameters())
    source_parameters = dict(source_module.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise ValueError("target_module and source_module must share the same parameter structure")

    for name, target_parameter in target_parameters.items():
        source_parameter = source_parameters[name]
        if target_parameter.shape != source_parameter.shape:
            raise ValueError(f"parameter shape mismatch for {name}")
        target_parameter.mul_(momentum).add_(source_parameter, alpha=1.0 - momentum)

    target_buffers = dict(target_module.named_buffers())
    source_buffers = dict(source_module.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise ValueError("target_module and source_module must share the same buffer structure")

    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.shape != source_buffer.shape:
            raise ValueError(f"buffer shape mismatch for {name}")
        target_buffer.copy_(source_buffer)
