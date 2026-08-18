import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # This zero-sized tensor exists only so device/dtype remain available on
        # otherwise parameterless modules. It never participates in forward, so
        # marking it trainable makes DDP wait for a gradient that cannot exist.
        self._dummy_variable = nn.Parameter(requires_grad=False)

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
