import torch


def get_device() -> torch.device:
    device_type = "cpu"
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    return torch.device(device_type)
