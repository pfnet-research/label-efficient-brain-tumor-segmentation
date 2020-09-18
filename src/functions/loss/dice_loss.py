import numpy as np
import chainer
from chainer import function
from chainer.backends import cuda
from chainer.utils import type_check, force_array
from chainer.functions.activation import softmax
from src.utils.encode_one_hot_vector import encode_one_hot_vector


class DiceLoss(function.Function):
    def __init__(self, eps=1e-7, weight=False, encode=True):
        self.eps = eps
        self.intersect = None
        self.union = None
        self.weighted_dice = weight
        self.encode = encode

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == np.int32,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[-3:] == t_type.shape[-3:],
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
        if self.encode:
            t = encode_one_hot_vector(t, x.shape[1])
        axis = (0,) + tuple(range(2, x.ndim))
        self.intersect = xp.sum((x * t), axis=axis)
        self.union = xp.sum((x * x), axis=axis) + xp.sum((t * t), axis=axis)
        dice = (2. * self.intersect + self.eps) / (self.union + self.eps)
        if self.weighted_dice:
            cw = xp.array([
                1., 1., 1., 1., 0.5, 0.5, 0.8, 0.8, 0.5, 0.8, 0.8,
                0.8, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 1., 1., 1.], dtype='float32')
            dice = dice*cw*x.shape[1]/xp.sum(cw)
        return force_array(xp.mean(1. - dice), dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        nb_class = x.shape[1]
        t = encode_one_hot_vector(t, nb_class)

        gx = xp.zeros_like(x)
        gloss = grad_outputs[0]
        cw = xp.array([
            1., 1., 1., 1., 0.5, 0.5, 0.8, 0.8, 0.5, 0.8, 0.8,
            0.8, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 1., 1., 1.], dtype='float32')
        for i, w in zip(range(nb_class), cw):
            x_i = x[:, i]
            t_i = t[:, i]
            intersect = self.intersect[i]
            union = self.union[i]

            numerator = xp.multiply(union + self.eps, t_i) - \
                xp.multiply(2. * intersect + self.eps, x_i)
            denominator = xp.power(union + self.eps, 2)
            d_dice = 2 * xp.divide(numerator, denominator).astype(xp.float32)
            if self.weighted_dice:
                gx[:, i] = d_dice*w*nb_class/xp.sum(cw)
            else:
                gx[:, i] = d_dice

        gx *= gloss / nb_class
        return -gx.astype(xp.float32), None


def dice_loss(x, t, eps=1e-7, weight=False, encode=True):
    return DiceLoss(eps, weight, encode)(x, t)


def softmax_dice_loss(x, t, eps=1e-7, weight=False, encode=True):
    x1 = softmax.softmax(x, axis=1)
    return dice_loss(x1, t, eps, weight, encode)
