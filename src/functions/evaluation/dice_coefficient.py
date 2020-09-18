from six import moves
import numpy as np

from chainer import function
from chainer import functions as F
from chainer.backends import cuda
from chainer.utils import type_check


class DiceCoefficient(function.Function):

    def __init__(self, ret_nan=True, dataset='task8hepatic', eps=1e-7, is_brats=False):
        self.ret_nan = ret_nan  # return NaN if union==0
        self.dataset = dataset
        self.eps = eps
        self.is_brats = is_brats

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == np.int32
        )

        t_ndim = type_check.eval(t_type.ndim)
        type_check.expect(
            x_type.ndim >= t_type.ndim,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
        )
        for i in moves.range(t_ndim + 1, type_check.eval(x_type.ndim)):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        """
        compute average Dice coefficient between two label images
        Math:
            DC = \frac{2|A\cap B|}{|A|+|B|}
        Args:
            inputs ((array_like, array_like)):
            Input pair (prediction, ground_truth)
        Returns:
            dice (float)
        """
        xp = cuda.get_array_module(*inputs)
        y, t = inputs
        number_class = y.shape[1]
        y = y.argmax(axis=1).reshape(t.shape)
        axis = tuple(range(t.ndim))

        dice = xp.zeros(number_class, dtype=xp.float32)
        for i in range(number_class):
            if self.is_brats:
                if i == 1:
                    # Enhancing tumor
                    y_match = xp.equal(y, 3).astype(xp.float32)
                    t_match = xp.equal(t, 3).astype(xp.float32)
                elif i == 2:
                    # Tumor core
                    y_match = (xp.equal(y, 2)+xp.equal(y, 3)).astype(xp.float32)
                    t_match = (xp.equal(t, 2)+xp.equal(t, 3)).astype(xp.float32)
                elif i == 3:
                    # Whole Tumor
                    y_match = (xp.equal(y, 1)+xp.equal(y, 2)+xp.equal(y, 3)).astype(xp.float32)
                    t_match = (xp.equal(t, 1)+xp.equal(t, 2)+xp.equal(t, 3)).astype(xp.float32)
                else:
                    y_match = xp.equal(y, i).astype(xp.float32)
                    t_match = xp.equal(t, i).astype(xp.float32)
            else:
                y_match = xp.equal(y, i).astype(xp.float32)
                t_match = xp.equal(t, i).astype(xp.float32)

            intersect = xp.sum(y_match * t_match, axis=axis)
            union = xp.sum(y_match, axis=axis) + xp.sum(t_match, axis=axis)
            if union == 0.:
                intersect += 0.5*self.eps
                union += self.eps
                dice[i] = 0.0
                if self.ret_nan:
                    dice[i] = np.nan
            else:
                dice[i] = 2.0 * (intersect / union)

        return xp.asarray(dice, dtype=xp.float32),


def dice_coefficient(y, t, ret_nan=False, dataset='task8hepatic', eps=1e-7, is_brats=False):
    return DiceCoefficient(ret_nan=ret_nan, dataset=dataset, eps=eps, is_brats=is_brats)(y, t)


def mean_dice_coefficient(dice_coefficients, ret_nan=True):
    if ret_nan:
        xp = cuda.get_array_module(dice_coefficients)
        selector = ~xp.isnan(dice_coefficients.data)
        dice_coefficients = F.get_item(dice_coefficients, selector)
    return F.mean(dice_coefficients, keepdims=True)
