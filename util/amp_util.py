import torch

class MaybeAMP:
    def __init__(self, device, enabled):
        self.device = device
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            self.amp = torch.amp.autocast(self.device.type)
            self.amp.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self.amp.__exit__(exc_type, exc_value, traceback)