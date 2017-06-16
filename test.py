import numpy
import pytest
import torch
from torch.autograd import Variable, gradcheck

from weight_norm import WeightNorm


device_host = [lambda x: x.cpu(), lambda x: x.cuda() if torch.cuda.is_available() else x]


@pytest.mark.parametrize("on", device_host)
def test_zero_initialized(on):
    linear = torch.nn.Linear(2, 3)
    linear.weight.data.fill_(0)
    linear.bias.data.fill_(0)
    linear = on(WeightNorm(linear, ["weight", "bias"]))
    xs = on(Variable(torch.randn(5, 2)))
    assert gradcheck(linear, (xs,))


@pytest.mark.parametrize("on", device_host)
def test_laziness(on):
    f = torch.nn.Linear(2, 3)
    f = WeightNorm(f, ["weight", "bias"])
    x = on(Variable(torch.randn(5, 2)))
    on(f)
    f.eval()
    f(x)
    wid1 = id(f.base.weight)
    _ = f(x)
    wid2 = id(f.base.weight)
    assert wid1 == wid2, "until trained, weight cannot be changed"

    f.train()
    _ = f(x)
    wid3 = id(f.base.weight)
    assert wid2 != wid3, "during training, weight should be changed"

    f.eval()
    _ = f(x)
    wid4 = id(f.base.weight)
    assert wid3 != wid4, "after training, weight should be changed for the first time"
    _ = f(x)
    wid5 = id(f.base.weight)
    assert wid4 == wid5, "after the first eval, weight should not be changed"


@pytest.mark.parametrize("on", device_host)
def test_gradcheck_linear(on):
    linear = on(WeightNorm(torch.nn.Linear(2, 3), ["weight", "bias"]))
    xs = on(Variable(torch.randn(5, 2)))
    assert gradcheck(linear, (xs,))


@pytest.mark.parametrize("on", device_host)
def test_gradcheck_conv(on):
    conv = on(WeightNorm(torch.nn.Conv1d(2, 3, 3), ["weight", "bias"]))
    xs = on(Variable(torch.randn(5, 2, 7)))
    assert gradcheck(conv, (xs,))


@pytest.mark.parametrize("on", device_host)
def test_serializable(on):
    linear = on(WeightNorm(torch.nn.Linear(2, 3), ["weight", "bias"]))
    torch.save(linear, "/tmp/linear.pkl")
    xs = on(Variable(torch.randn(5, 2)))
    linear_load = torch.load("/tmp/linear.pkl")
    numpy.testing.assert_allclose(linear(xs).cpu().data.numpy(), linear_load(xs).cpu().data.numpy())


class AutogradWeightNorm(WeightNorm):
    def update(self, name, name_g, name_v):
        g = getattr(self, name_g)
        v = getattr(self, name_v)
        w = v * (g / (torch.norm(v) + self.eps)).expand_as(v)
        setattr(self.base, name, w)


@pytest.mark.parametrize("on", device_host)
def test_autograd(on):
    linear = torch.nn.Linear(2, 3)
    f = on(WeightNorm(linear, ["weight", "bias"]))
    g = on(AutogradWeightNorm(linear, ["weight", "bias"]))
    f.train()
    g.train()

    x = on(Variable(torch.randn(5, 2), requires_grad=True))
    fx = f(x)
    gx = g(x)
    numpy.testing.assert_allclose(fx.cpu().data.numpy(), gx.cpu().data.numpy(), rtol=1e-6, atol=1e-4)

    fx.sum().backward()
    gx.sum().backward()
    for name in f.name_g_list + f.name_v_list:
        df = getattr(f, name).grad.data.cpu().numpy()
        dg = getattr(g, name).grad.data.cpu().numpy()
        numpy.testing.assert_allclose(df, dg, rtol=1e-6, atol=1e-4)
