import chainer.functions as F
from src.functions.loss import softmax_dice_loss
from src.functions.loss import focal_loss
import cupy


def struct_dice_loss_plus_cross_entropy(x, t, w=0.5):
    cw = cupy.array([
        1., 1., 1., 1., 0.5, 0.5, 0.8, 0.8, 0.5, 0.8, 0.8,
        0.8, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 1., 1., 1.], dtype='float32')
    return w * softmax_dice_loss(x, t) + (1 - w) * F.softmax_cross_entropy(x, t, class_weight=cw)


def dice_loss_plus_cross_entropy(x, t, w=0.5, encode=True):
    return w * softmax_dice_loss(x, t, encode=encode) + (1 - w) * F.softmax_cross_entropy(x, t)


def dice_loss_plus_focal_loss(x, t, w=0.5):
    return w * softmax_dice_loss(x, t) + (1 - w) * focal_loss(x, t)
