import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class ResizeImages3D(function_node.FunctionNode):

    def __init__(self, output_shape):
        self.out_H = output_shape[0]
        self.out_W = output_shape[1]
        self.out_D = output_shape[2]

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.char == 'f',
            x_type.ndim == 5
        )

    def forward(self, inputs):
        x, = inputs
        xp = cuda.get_array_module(x)

        B, C, H, W, D = x.shape

        v_1d = xp.linspace(0, H - 1, num=self.out_H)
        u_1d = xp.linspace(0, W - 1, num=self.out_W)
        t_1d = xp.linspace(0, D - 1, num=self.out_D)
        grid = xp.meshgrid(v_1d, u_1d, t_1d, indexing='ij')
        v = grid[0].ravel()
        u = grid[1].ravel()
        t = grid[2].ravel()

        v0 = xp.floor(v).astype(numpy.int32)
        v0 = v0.clip(0, H - 2)
        u0 = xp.floor(u).astype(numpy.int32)
        u0 = u0.clip(0, W - 2)
        t0 = xp.floor(t).astype(numpy.int32)
        t0 = t0.clip(0, D - 2)

        y = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    wv = xp.abs(v0 + (1 - i) - v)
                    wu = xp.abs(u0 + (1 - j) - u)
                    wt = xp.abs(t0 + (1 - k) - t)
                    w = (wv * wu * wt).astype(x.dtype, copy=False)
                    y += w[None, None, :] * x[:, :, v0 + i, u0 + j, t0 + k]
        y = y.reshape(B, C, self.out_H, self.out_W, self.out_D)
        return y,

    def backward(self, indexes, grad_outputs):
        return ResizeImagesGrad3D(
            self.inputs[0].shape,
            (self.out_H, self.out_W, self.out_D)).apply(grad_outputs)


class ResizeImagesGrad3D(function_node.FunctionNode):

    def __init__(self, input_shape, output_shape):
        self.out_H = output_shape[0]
        self.out_W = output_shape[1]
        self.out_D = output_shape[2]
        self.input_shape = input_shape

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)

        x_type = in_types[0]
        type_check.expect(
            x_type.dtype.char == 'f',
            x_type.ndim == 5
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        gy, = inputs

        B, C, H, W, D = self.input_shape

        v_1d = xp.linspace(0, H - 1, num=self.out_H)
        u_1d = xp.linspace(0, W - 1, num=self.out_W)
        t_1d = xp.linspace(0, D - 1, num=self.out_D)
        grid = xp.meshgrid(v_1d, u_1d, t_1d, indexing='ij')
        v = grid[0].ravel()
        u = grid[1].ravel()
        t = grid[2].ravel()

        v0 = xp.floor(v).astype(numpy.int32)
        v0 = v0.clip(0, H - 2)
        u0 = xp.floor(u).astype(numpy.int32)
        u0 = u0.clip(0, W - 2)
        t0 = xp.floor(t).astype(numpy.int32)
        t0 = t0.clip(0, D - 2)

        if xp is numpy:
            scatter_add = numpy.add.at
        else:
            scatter_add = cuda.cupyx.scatter_add

        gx = xp.zeros(self.input_shape, dtype=gy.dtype)
        gy = gy.reshape(B, C, -1)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    wv = xp.abs(v0 + (1 - i) - v)
                    wu = xp.abs(u0 + (1 - j) - u)
                    wt = xp.abs(t0 + (1 - k) - t)
                    w = (wv * wu * wt).astype(gy.dtype, copy=False)
                    scatter_add(
                        gx,
                        (slice(None), slice(None), v0 + i, u0 + j, t0 + k),
                        gy * w)
        return gx,

    def backward(self, indexes, grad_outputs):
        return ResizeImages3D(
            (self.out_H, self.out_W, self.out_D)).apply(grad_outputs)


def resize_images_3d(x, output_shape):
    """Resize images to the given shape.
    This function resizes 3D data to :obj:`output_shape`.
    Currently, only bilinear interpolation is supported as the sampling method.
    Notation: here is a notation for dimensionalities.
    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h`, :math:`w` and :math:`d` are the height, width and depth of the
        input image, respectively.
    - :math:`h_O`, :math:`w_O` and :math:`d_O` are the height, width and depth
        of the output image.
    Args:
        x (~chainer.Variable):
            Input variable of shape :math:`(n, c_I, h, w, d)`.
        output_shape (tuple):
            This is a tuple of length 3 whose values are :obj:`(h_O, w_O, d_O)`
    Returns:
        ~chainer.Variable: Resized image whose shape is \
            :math:`(n, c_I, h_O, w_O, d_O)`.
    """
    return ResizeImages3D(output_shape).apply((x,))[0]
