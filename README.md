# Pytorch Weight Normalization

+ ref https://arxiv.org/pdf/1602.07868v3.pdf
+ forked from https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f


## example

``` python
import torch
from torch.autograd import Variable
from weight_norm import WeightNorm

# weight-normalize linear.weight and linear.bias
linear = WeightNorm(torch.nn.Linear(2, 3), ["weight", "bias"])
y = linear(Variable(torch.randn(5, 2)))  # used as same as Linear

# weight-normalize conv.weight and conv.bias
conv = WeightNorm(torch.nn.Conv1d(2, 3, 3), ["weight", "bias"])
y = conv(Variable(torch.randn(5, 2, 7)))  # used as same as Conv1d
```


## tested parts

+ gradient check (numerical and autograd comparison)
+ serializable
+ avoid zero division error
+ lazy at eval()
