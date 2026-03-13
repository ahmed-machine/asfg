import torch

_device = None

def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device

def set_device(d):
    global _device
    _device = torch.device(d) if isinstance(d, str) else d
