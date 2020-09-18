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

        u_1d = xp.linspace(0, W - 1, num=self.out_W)
        v_1d = xp.linspace(0, H - 1, num=self.out_H)
        t_1d = xp.linspace(0, D - 1, num=self.out_D)
        grid = xp.meshgrid(u_1d, v_1d, t_1d)
        u = grid[0].ravel()
        v = grid[1].ravel()
        t = grid[2].ravel()

        u0 = xp.floor(u).astype(numpy.int32)
        u0 = u0.clip(0, W - 2)
        u1 = u0 + 1
        v0 = xp.floor(v).astype(numpy.int32)
        v0 = v0.clip(0, H - 2)
        v1 = v0 + 1
        t0 = xp.floor(t).astype(numpy.int32)
        t0 = t0.clip(0, D - 2)
        t1 = t0 + 1

        # weights
        w1 = (u1 - u) * (v1 - v) * (t1 - t)
        w2 = (u - u0) * (v1 - v) * (t1 - t)
        w3 = (u1 - u) * (v - v0) * (t1 - t)
        w4 = (u - u0) * (v - v0) * (t1 - t)
        w5 = (u1 - u) * (v1 - v) * (t - t0)
        w6 = (u - u0) * (v1 - v) * (t - t0)
        w7 = (u1 - u) * (v - v0) * (t - t0)
        w8 = (u - u0) * (v - v0) * (t - t0)
        w1 = w1.astype(x.dtype)
        w2 = w2.astype(x.dtype)
        w3 = w3.astype(x.dtype)
        w4 = w4.astype(x.dtype)
        w5 = w5.astype(x.dtype)
        w6 = w6.astype(x.dtype)
        w7 = w7.astype(x.dtype)
        w8 = w8.astype(x.dtype)

        y = (w1[None, None, :] * x[:, :, v0, u0, t0] +
             w2[None, None, :] * x[:, :, v0, u1, t0] +
             w3[None, None, :] * x[:, :, v1, u0, t0] +
             w4[None, None, :] * x[:, :, v1, u1, t0] +
             w5[None, None, :] * x[:, :, v0, u0, t1] +
             w6[None, None, :] * x[:, :, v0, u1, t1] +
             w7[None, None, :] * x[:, :, v1, u0, t1] +
             w8[None, None, :] * x[:, :, v1, u1, t1])
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

        u_1d = xp.linspace(0, W - 1, num=self.out_W)
        v_1d = xp.linspace(0, H - 1, num=self.out_H)
        t_1d = xp.linspace(0, D - 1, num=self.out_D)
        grid = xp.meshgrid(u_1d, v_1d, t_1d)
        u = grid[0].ravel()
        v = grid[1].ravel()
        t = grid[2].ravel()

        u0 = xp.floor(u).astype(numpy.int32)
        u0 = u0.clip(0, W - 2)
        u1 = u0 + 1
        v0 = xp.floor(v).astype(numpy.int32)
        v0 = v0.clip(0, H - 2)
        v1 = v0 + 1
        t0 = xp.floor(t).astype(numpy.int32)
        t0 = t0.clip(0, D - 2)
        t1 = t0 + 1

        # weights
        wu0 = u - u0
        wu1 = u1 - u
        wv0 = v - v0
        wv1 = v1 - v
        wt0 = t - t0
        wt1 = t1 - t
        wu0 = wu0.astype(gy.dtype)
        wu1 = wu1.astype(gy.dtype)
        wv0 = wv0.astype(gy.dtype)
        wv1 = wv1.astype(gy.dtype)
        wt0 = wt0.astype(gy.dtype)
        wt1 = wt1.astype(gy.dtype)

        # --- gx
        if xp is numpy:
            scatter_add = numpy.add.at
        else:
            scatter_add = cuda.cupyx.scatter_add

        gx = xp.zeros(self.input_shape, dtype=gy.dtype)
        gy = gy.reshape(B, C, -1)
        scatter_add(gx, (slice(None), slice(None), v0, u0, t0),
                    gy * wu1 * wv1 * wt1)
        scatter_add(gx, (slice(None), slice(None), v0, u1, t0),
                    gy * wu0 * wv1 * wt1)
        scatter_add(gx, (slice(None), slice(None), v1, u0, t0),
                    gy * wu1 * wv0 * wt1)
        scatter_add(gx, (slice(None), slice(None), v1, u1, t0),
                    gy * wu0 * wv0 * wt1)
        scatter_add(gx, (slice(None), slice(None), v0, u0, t1),
                    gy * wu1 * wv1 * wt0)
        scatter_add(gx, (slice(None), slice(None), v0, u1, t1),
                    gy * wu0 * wv1 * wt0)
        scatter_add(gx, (slice(None), slice(None), v1, u0, t1),
                    gy * wu1 * wv0 * wt0)
        scatter_add(gx, (slice(None), slice(None), v1, u1, t1),
                    gy * wu0 * wv0 * wt0)
        return gx,

    def backward(self, indexes, grad_outputs):
        return ResizeImages3D(
            (self.out_H, self.out_W, self.out_D)).apply(grad_outputs)


def resize_images_3d(x, output_shape):
    """Resize images to the given shape.
    This function resizes 3D data to :obj:`output_shape`.
    Currently, only bilinear interpolation is supported as the sampling method.
    Notatition: here is a notation for dimensionalities.
    - :math:`n` is the batch size.
    - :math:`c_I` is the number of the input channels.
    - :math:`h`, :math:`w` and :math:`d` are the height, width and depth of the
        input image, respectively.
    - :math:`h_O`, :math:`w_O` and :math:`d_0` are the height, width and depth
        of the output image.
    Args:
        x (~chainer.Variable):
        Input variable of shape :math:`(n, c_I, h, w, d)`.
        output_shape (tuple):
        This is a tuple of length 3 whose values are :obj:`(h_O, w_O, d_O)`.
    Returns:
        ~chainer.Variable: Resized image whose shape is \
            :math:`(n, c_I, h_O, w_O, d_O)`.
    """
    return ResizeImages3D(output_shape).apply((x,))[0]
