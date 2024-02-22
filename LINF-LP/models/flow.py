import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from models import register


@register('flow')
class Flow(nn.Module):
    def __init__(self, flow_layers=10, patch_size=1, name='flow'):
        super(Flow, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        self.n_layers = flow_layers
        self.ps_square = patch_size*patch_size

        self.linears = torch.nn.ModuleList([NaiveLinear(3*patch_size*patch_size) for _ in range(flow_layers)])
        self.last = NaiveLinear(3*patch_size*patch_size)

        self.log2pi = float(np.log(2 * np.pi))
        self.affine_eps = 0.0001

    def get_logdet(self, scale):
        return torch.sum(torch.log(scale), dim=-1)

    def affine_forward(self, inputs, affine_info):
        scale, shift = torch.split(affine_info, 3*self.ps_square, dim=-1)
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        outputs = inputs * scale + shift
        logabsdet = self.get_logdet(scale)
        return outputs, logabsdet
        
    def affine_inverse(self, inputs, affine_info):
        scale, shift = torch.split(affine_info, 3*self.ps_square, dim=-1)
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        outputs = (inputs - shift) / scale
        return outputs

    def forward(self, x, affine_info):
        z, total_log_det_J = x, x.new_zeros(x.shape[0])
        for i in range(self.n_layers):
            z, log_det_J = self.linears[i](z)
            total_log_det_J += log_det_J
            z, log_det_J = self.affine_forward(z, affine_info[:, i*6*self.ps_square:i*6*self.ps_square+6*self.ps_square])
            total_log_det_J += log_det_J
        z, log_det_J = self.last(z)
        total_log_det_J += log_det_J
        # add base distribution log_prob
        total_log_det_J += torch.sum(-0.5 * (z ** 2 + self.log2pi), -1)
        return z, total_log_det_J

    def inverse(self, z, affine_info):
        x = z
        x = self.last.inverse(x)
        for i in reversed(range(self.n_layers)):
            x = self.affine_inverse(x, affine_info[:, i*6*self.ps_square:i*6*self.ps_square+6*self.ps_square])
            x = self.linears[i].inverse(x)
        return x


def get_logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


class NaiveLinear(nn.Module):
    """A general linear transform that uses an unconstrained weight matrix.
    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.
    Both forward and inverse directions have a cost of O(D^3), where D is the dimension
    of the input.
    """

    def __init__(self, features=3):
        """Constructor.
        Args:
            features: int, number of input features.
        Raises:
            TypeError: if `features` is not a positive integer.
        """
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(features))

        self._weight = nn.Parameter(torch.empty(features, features))
        stdv = 1.0 / np.sqrt(8)
        init.uniform_(self._weight, -stdv, stdv)

    def forward(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        # y = x * A.t + b
        outputs = F.linear(inputs, self._weight, self.bias)
        logabsdet = get_logabsdet(self._weight)
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def inverse(self, inputs):
        """Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        outputs = inputs - self.bias
        outputs = torch.linalg.solve(self._weight, outputs.t())  # Linear-system solver.
        outputs = outputs.t()
        return outputs

    def logabsdet(self):
        """Cost:
            logabsdet = O(D^3)
        where:
            D = num of features
        """
        return get_logabsdet(self._weight)


if __name__ == '__main__':
    net = NaiveLinear(3*9)
    print("forward: RGB -> Z")
    rgb = torch.rand(4, 3*9)    # model 3*3 patch
    z, logabsdet = net(rgb)
    print(logabsdet)
    print("backward: Z -> RGB")
    rgb_inv = net.inverse(z)
    print(torch.mean(torch.abs(rgb_inv - rgb)))
