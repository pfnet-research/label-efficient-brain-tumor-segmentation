import numpy as np
from typing import Tuple

from chainer.backends import cuda
from chainer.types import NdArray


def resize_images_3d_nearest_neighbor(x: NdArray, output_shape: Tuple[int, int, int]) -> NdArray:
    # Note: this function is not differentiable
    # Resize from (H, W, D) to (H', W', D')
    out_H, out_W, out_D = output_shape
    xp = cuda.get_array_module(x)

    B, C, H, W, D = x.shape
    v_1d = xp.linspace(0, H - 1, num=out_H)
    u_1d = xp.linspace(0, W - 1, num=out_W)
    t_1d = xp.linspace(0, D - 1, num=out_D)
    grid = xp.meshgrid(v_1d, u_1d, t_1d, indexing='ij')
    v = grid[0].ravel()  # (H'W'D',)
    u = grid[1].ravel()
    t = grid[2].ravel()

    v0 = xp.floor(v).astype(np.int32)
    v0 = v0.clip(0, H - 2)
    u0 = xp.floor(u).astype(np.int32)
    u0 = u0.clip(0, W - 2)
    t0 = xp.floor(t).astype(np.int32)
    t0 = t0.clip(0, D - 2)

    ws, xs = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                wv = xp.abs(v0 + (1 - i) - v)
                wu = xp.abs(u0 + (1 - j) - u)
                wt = xp.abs(t0 + (1 - k) - t)
                w = wv * wu * wt
                ws.append(w)
                xs.append(x[:, :, v0 + i, u0 + j, t0 + k])
    ws = xp.stack(ws)  # (8, H'W'D')
    xs = xp.stack(xs)  # (8, B, C, H'W'D')
    xs = xs.transpose((0, 3, 1, 2))  # (8, H'W'D', B, C)

    target_indices = xp.argmax(ws, axis=0)  # (H'W'D',)
    y = xs[target_indices, np.arange(out_H * out_W * out_D)]  # (H'W'D', B, C)
    y = y.transpose((1, 2, 0)).reshape((B, C, out_H, out_W, out_D))  # (B, C, H', W', D')
    return y
