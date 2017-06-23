import numpy
import pytest
import torch
from torch.autograd import Variable, gradcheck

from weight_norm import WeightNorm


def to_cpu(x):
    return x.cpu()


def to_gpu(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


device_host = [to_cpu, to_gpu]


@pytest.mark.parametrize("on", device_host)
def test_zero_initialized(on):
    linear = torch.nn.Linear(2, 3)
    linear.weight.data.fill_(0)
    linear.bias.data.fill_(0)
    linear = on(WeightNorm(linear, ["weight", "bias"]))
    xs = on(Variable(torch.randn(5, 2)))
    linear(xs)


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


m_x = [(torch.nn.Linear(2, 3), torch.randn(5, 2)),
       (torch.nn.Conv1d(2, 3, 3), torch.randn(5, 2, 7)),
       (torch.nn.LSTM(2, 3), torch.randn(4, 5, 2))]


@pytest.mark.parametrize("on", device_host)
def test_serializable(on):
    for m, x in m_x:
        w = on(WeightNorm(m))
        torch.save(w, "/tmp/w.pkl")
        xs = on(Variable(x))
        w_load = torch.load("/tmp/w.pkl")
        wxs = w(xs)
        if isinstance(wxs, tuple):
            wxs = wxs[0]
        vxs = w_load(xs)
        if isinstance(vxs, tuple):
            vxs = vxs[0]
        for wx, vx in zip(wxs, vxs):
            numpy.testing.assert_allclose(wx.cpu().data.numpy(), vx.cpu().data.numpy())


class AutogradWeightNorm(WeightNorm):
    def update(self, name, name_g, name_v):
        g = getattr(self, name_g)
        v = getattr(self, name_v)
        w = v * (g / (torch.norm(v) + self.eps)).expand_as(v)
        setattr(self.base, name, w)


@pytest.mark.parametrize("on", device_host)
def test_autograd(on):
    for m, x in m_x:
        on(m)
        f = (WeightNorm(m))
        g = (AutogradWeightNorm(m, f.name_list))
        f.train()
        g.train()
        on(f)
        on(g)

        x = on(Variable(x, requires_grad=True))
        fx = f(x)
        if isinstance(fx, tuple):
            fx = fx[0]
        gx = g(x)
        if isinstance(gx, tuple):
            gx = gx[0]
        numpy.testing.assert_allclose(fx.cpu().data.numpy(), gx.cpu().data.numpy(), rtol=1e-6, atol=1e-4)

        print(f.weight_v)
        fx.sum().backward(retain_variables=True)
        gx.sum().backward(retain_variables=True)
        for name in f.name_g_list + f.name_v_list:
            df = getattr(f, name).grad.data.cpu().numpy()
            dg = getattr(g, name).grad.data.cpu().numpy()
            numpy.testing.assert_allclose(df, dg, rtol=1e-6, atol=1e-4)

