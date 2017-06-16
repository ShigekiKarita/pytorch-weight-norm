# forked from https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f
# TODO: test like https://github.com/soskek/weight_normalization/tree/b60b81e19c2e9eaaff99810c4072d5b475f01ba4


from typing import List

import numpy
import torch
from torch.nn import Parameter, Module


class WeightNorm(Module):
    def __init__(self, base: Module, name_list: List[str], eps: float=float(numpy.finfo(numpy.float32).eps)):
        super().__init__()
        self.base = base
        self.eps = eps
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

            # remove w from parameter list
            delattr(self.base, name)

            # add g and v as new parameters
            self.register_parameter(name_g, g)
            self.register_parameter(name_v, v)
            self.name_g_list.append(name_g)
            self.name_v_list.append(name_v)

    def forward(self, *args, **kwargs):
        for name, name_g, name_v in zip(self.name_list, self.name_g_list, self.name_v_list):
            g = getattr(self, name_g)
            v = getattr(self, name_v)
            w = v * (g / (torch.norm(v) + self.eps)).expand_as(v)
            setattr(self.base, name, w)

        return self.base(*args, **kwargs)


if __name__ == '__main__':
    # gradient check
    linear = torch.nn.Linear(2, 3)
    linear = WeightNorm(linear, ["weight", "bias"])
    xs = torch.autograd.Variable(torch.randn(5, 2))
    torch.autograd.gradcheck(linear, (xs,), eps=1e-6, atol=1e-4)

    # pickle check
    torch.save(linear, "/tmp/linear.pkl")
    linear_load = torch.load("/tmp/linear.pkl")
    numpy.testing.assert_allclose(linear(xs).data.numpy(), linear_load(xs).data.numpy())
