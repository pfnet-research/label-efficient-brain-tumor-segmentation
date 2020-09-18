import chainer.functions as F
from chainer.backends.cuda import get_array_module


def focal_loss(x, t, gamma=0.5, eps=1e-6):
    xp = get_array_module(t)

    p = F.softmax(x)
    p = F.clip(p, x_min=eps, x_max=1-eps)
    log_p = F.log_softmax(x)
    t_onehot = xp.eye(x.shape[1])[t.ravel()]

    loss_sce = -1 * t_onehot * log_p
    loss_focal = F.sum(loss_sce * (1. - p) ** gamma, axis=1)

    return F.mean(loss_focal)
