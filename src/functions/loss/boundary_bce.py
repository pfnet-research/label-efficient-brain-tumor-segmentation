import chainer
from chainer import function
from chainer.backends import cuda
from chainer.utils import type_check, force_array
from chainer.functions import softmax_cross_entropy


class BoundaryBCE(function.Function):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[2:],
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
        num_class = t.shape[1]
        bound_loss = 0.
        for i in range(1, num_class):
            # convert label to a (0,1)label for each class
            tt = t[0, i, :, :, :].astype(xp.int32)
            tt = tt.reshape(-1)
            beta = xp.sum(tt)/tt.size
            class_weight = xp.stack([1-beta, beta])

            xx = x[:, [0, i], :, :, :]
            xx = xx.reshape(-1, 2)
            bound_loss += softmax_cross_entropy(xx, tt, class_weight=class_weight)
        return force_array(bound_loss.data, ),


def boundary_bce(x, t, eps=1e-7):
    return BoundaryBCE(eps)(x, t)
