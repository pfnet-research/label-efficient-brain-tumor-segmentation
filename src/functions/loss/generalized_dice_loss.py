import numpy as np
import chainer
from chainer import function
from chainer.backends import cuda
from chainer.utils import type_check, force_array
from chainer.functions.activation import softmax
from src.utils.encode_one_hot_vector import encode_one_hot_vector


class GeneralizedDiceLoss(function.Function):
    def __init__(self, eps=1e-7):
        # avoid zero division error
        self.eps = eps
        self.w = None
        self.intersect = None
        self.union = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == np.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    @staticmethod
    def _check_input_values(x, t):
        if not (((0 <= t) &
                 (t < x.shape[1]))).all():
            msg = ('Each label `t` need to satisfy '
                   '`0 <= t < x.shape[1]`')
            raise ValueError(msg)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)
        # one-hot encoding of ground truth
        t = encode_one_hot_vector(t, x.shape[1])
        # compute weight, intersection, and union
        axis = (0,) + tuple(range(2, x.ndim))
        sum_t = xp.sum(t, axis=axis)
        # avoid zero division error
        sum_t[sum_t == 0] = 1
        self.w = 1. / (sum_t*sum_t)
        self.intersect = xp.multiply(xp.sum((x * t), axis=axis), self.w)
        self.union = xp.multiply(
            xp.sum(x, axis=axis) + xp.sum(t, axis=axis), self.w)
        # compute dice loss
        dice = (2. * self.intersect + self.eps) / (self.union + self.eps)
        return force_array(1. - dice, dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        nb_class = x.shape[1]
        t = encode_one_hot_vector(t, nb_class)

        gx = xp.zeros_like(x)
        gloss = grad_outputs[0]
        for i in range(nb_class):
            t_i = t[:, i]
            intersect = self.intersect
            union = self.union
            w = self.w[i]
            numerator = xp.multiply(
                union + self.eps, t_i) - intersect + self.eps
            denominator = xp.power(union + self.eps, 2)
            d_dice = 2 * xp.divide(
                w * numerator, denominator).astype(xp.float32)
            gx[:, i] = d_dice

        gx *= gloss
        return -gx.astype(xp.float32), None


def generalized_dice_loss(x, t, eps=1e-7):
    return GeneralizedDiceLoss(eps)(x, t)


def softmax_generalized_dice_loss(x, t, eps=1e-7):
    x1 = softmax.softmax(x, axis=1)
    return generalized_dice_loss(x1, t, eps)
