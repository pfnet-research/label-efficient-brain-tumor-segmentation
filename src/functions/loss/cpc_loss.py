import numpy as np
import chainer
from chainer import function
import chainer.functions as F
from chainer.backends import cuda
from chainer.utils import type_check, force_array
import random


class CPCLoss(function.Function):
    """ Contrastive loss from Contrastive Predictive Coding,
    arXiv:1905.09272 """
    def __init__(self, upper=True, cpc_pattern='updown', num_neg=20, eps=1e-7):
        self.eps = eps
        # number of negative samples
        self.num_neg = num_neg
        self.upper = upper
        self.cpc_pattern = cpc_pattern

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'f',
            x_type.shape[0] == t_type.shape[0]
        )

    @staticmethod
    def _check_input_values(x, t):
        if not (0 <= t):
            msg = ('Each label `t` need to satisfy '
                   '`0 <= t `')
            raise ValueError(msg)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        b, c, gl0, gl1, gl2 = t.shape  # gl: grid length
        cut_l = int(gl2/2)

        if chainer.is_debug():
            self._check_input_values(x, t)

        if self.cpc_pattern == 'ichimatsu':
            correct_z = t[:, :, 2:gl0+2:4, 2:gl1+2:4, 1:gl2:2]
            neg_z = t[:, :, :, :, 0:gl2:2]
        else:
            if self.upper:
                correct_z = t[:, :, :, :, -cut_l:]
                neg_z = t[:, :, :, :, :-cut_l]
            else:
                correct_z = t[:, :, :, :, :cut_l]
                neg_z = t[:, :, :, :, cut_l:]

        if self.cpc_pattern == 'ichimatsu':
            xx = F.reshape(x, (c, -1))
            correct_z = F.reshape(correct_z, (c, -1))
            neg_z = F.reshape(neg_z, (c, -1))
            # selecting num_neg negative samples to concat with correct samples of z
            selection = random.sample(set(np.arange(neg_z.shape[1])), self.num_neg)
        else:
            xx = F.reshape(x, (c, gl0*gl1*cut_l))
            correct_z = F.reshape(correct_z, (c, gl0*gl1*cut_l))
            neg_z = F.reshape(neg_z, (c, gl0*gl1*(gl2-cut_l)))
            # selecting num_neg negative samples to concat with correct samples of z
            selection = random.sample(set(np.arange(gl0*gl1*(gl2-cut_l))), self.num_neg)

        z_hat_T = F.transpose(xx)
        # selecting num_neg negative samples to concat with correct samples of z
        selection = random.sample(set(np.arange(gl0*gl1*(gl2-cut_l))), self.num_neg)
        neg_z = neg_z[:, selection]
        z = F.concat((correct_z, neg_z), axis=1)

        ip = xp.dot(z_hat_T.data, z.data)
        if self.cpc_pattern == 'ichimatsu':
            cpc = F.softmax_cross_entropy(ip, xp.arange(12), reduce='mean')
        else:
            cpc = F.softmax_cross_entropy(ip, xp.arange(gl0*gl1*cut_l), reduce='mean')
        return force_array(xp.mean(cpc.data), dtype=xp.float32),


def cpc_loss(x, t, upper=True, cpc_pattern='updown', num_neg=20, eps=1e-7):
    return CPCLoss(upper, cpc_pattern, num_neg, eps)(x, t)
