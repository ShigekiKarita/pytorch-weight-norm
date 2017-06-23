from typing import List, Tuple

import numpy
import torch
from torch.autograd import Function
from torch.nn import Parameter, Module


class WeightNormOne(Function):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps
        self.norm = 0.0

    def forward(self, g: torch.FloatTensor, v: torch.FloatTensor) -> torch.FloatTensor:
        self.norm = torch.norm(v) + self.eps  # float cannot be saved for backward
        self.save_for_backward(g, v)
        return v * (g / (self.norm)).expand_as(v)

    def backward(self, dw: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        g, v = self.saved_tensors
        dg = dw.dot(v) / self.norm
        dv = dw * (g / self.norm).expand_as(dw)
        dv -= v * (g * dg / self.norm).expand_as(v)
        return dw.new([dg]), dv


class WeightNorm(Module):
    def __init__(self, base: Module, name_list: List[str]=None, eps: float = float(numpy.finfo(numpy.float32).eps)):
        super().__init__()
        if name_list is None:
            name_list = list(dict(base.named_parameters()).keys())
        self.base = base
        self.eps = eps
        self.trained = False
        self.name_list = name_list
        self.name_g_list = []
        self.name_v_list = []
        for name in name_list:
            param = getattr(self.base, name)
            g = torch.norm(param) + self.eps
            v = param / (g.expand_as(param))
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name + '_g'
            name_v = name + '_v'

            # add g and v as new parameters
            self.register_parameter(name_g, g)
            self.register_parameter(name_v, v)
            self.name_g_list.append(name_g)
            self.name_v_list.append(name_v)

            # init w (change Parameter to Variable)
            delattr(self.base, name)
            self.update(name, name_g, name_v)

    def cuda(self, device_id=None):
        for name in self.name_list:
            setattr(self.base, name, getattr(self.base, name).cuda(device_id=device_id))
        return super().cuda(device_id=device_id)

    def cpu(self, device_id=None):
        for name in self.name_list:
            setattr(self.base, name, getattr(self.base, name).cpu())  # Variable has no device id
        return super().cpu(device_id=device_id)

    def update(self, name: str, name_g: str, name_v: str):
        g = getattr(self, name_g)
        v = getattr(self, name_v)
        w = WeightNormOne(self.eps)(g, v)
        setattr(self.base, name, w)

    def forward(self, *args, **kwargs):
        if self.training or self.trained:
            for name, name_g, name_v in zip(self.name_list, self.name_g_list, self.name_v_list):
                self.update(name, name_g, name_v)
            self.trained = self.training

        return self.base(*args, **kwargs)
