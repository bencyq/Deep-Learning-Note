import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import torch
import sys
sys.path.append('../..')
from utils.accuracy import binary_accuracy



class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff):
        ctx.coeff = coeff
        output = input*1.0
        return output

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg()*ctx.coeff, None



class GradientReverseLayer(nn.Module):

    def __int__(self):
        super(GradientReverseLayer, self).__int__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)



class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha, lo=0.0, hi=1.0, max_iters=1000, auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        coeff = np.float(2*(self.hi - self.lo) / (1.0+ np.exp(-self.alpha*self.iter_num/self.max_iters))) - (self.hi - self.lo) + self.lo
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1



class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator, grl=None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discirminator = domain_discriminator
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight, reduction='mean')
        self.domain_discriminator_accuracy = None
    def forward(self, f_s, f_t, w_s=None, w_t=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discirminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5*(self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))
