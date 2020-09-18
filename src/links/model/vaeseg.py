import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links import GroupNormalization
from src.functions.array import resize_images_3d
from chainer.utils.conv_nd import im2col_nd


class CNR(chainer.Chain):
    """Convolution, Normalize, then ReLU"""

    def __init__(
            self,
            channels,
            norm=GroupNormalization,
            down_sampling=False,
            comm=None
    ):
        super(CNR, self).__init__()
        with self.init_scope():
            if down_sampling:
                self.c = L.Convolution3D(None, channels, 3, 2, 1)
            else:
                self.c = L.Convolution3D(None, channels, 3, 1, 1)
            if norm.__name__ == 'MultiNodeBatchNormalization':
                self.n = norm(channels, comm, eps=1e-5)
            elif norm.__name__ == 'BatchNormalization':
                self.n = norm(channels, eps=1e-5)
            elif norm.__name__ == 'GroupNormalization':
                self.n = norm(groups=8, size=channels)
            else:
                self.n = norm(channels)

    def forward(self, x):
        h = F.relu(self.n(self.c(x)))
        return h


class NRC(chainer.Chain):
    """Normalize, ReLU, then Convolution"""
    def __init__(
            self,
            in_channels,
            out_channels,
            norm=GroupNormalization,
            down_sampling=False,
            comm=None
    ):
        super(NRC, self).__init__()
        with self.init_scope():
            if norm.__name__ == 'MultiNodeBatchNormalization':
                self.n = norm(in_channels, comm, eps=1e-5)
            elif norm.__name__ == 'BatchNormalization':
                self.n = norm(in_channels, eps=1e-5)
            elif norm.__name__ == 'GroupNormalization':
                self.n = norm(groups=8, size=in_channels)
            else:
                self.n = norm(in_channels)
            if down_sampling:
                self.c = L.Convolution3D(None, out_channels, 3, 2, 1)
            else:
                self.c = L.Convolution3D(None, out_channels, 3, 1, 1)

    def forward(self, x):
        h = self.c(F.relu(self.n(x)))
        return h


class ResBlock(chainer.Chain):

    def __init__(
            self,
            channels,
            norm=GroupNormalization,
            bn_first=True,
            comm=None,
            concat_mode=False
    ):
        super(ResBlock, self).__init__()
        with self.init_scope():
            if bn_first:
                if concat_mode:
                    channels *= 2
                self.block1 = NRC(channels, channels, norm, False, comm)
                self.block2 = NRC(channels, channels, norm, False, comm)
            else:
                self.block1 = CNR(channels, norm, False, comm)
                self.block2 = CNR(channels, norm, False, comm)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + x


class DownBlock(chainer.Chain):
    """down sample (conv stride2), then ResBlock * num of layers"""

    def __init__(
            self,
            channels,
            norm=GroupNormalization,
            down_sample=True,
            n_blocks=1,
            bn_first=True,
            comm=None
    ):
        self.down_sample = down_sample
        self.n_blocks = n_blocks
        super(DownBlock, self).__init__()
        with self.init_scope():
            if down_sample:
                self.d = L.Convolution3D(None, channels, 3, 2, 1)
            else:
                self.d = L.Convolution3D(None, channels, 3, 1, 1)
            for i in range(n_blocks):
                layer = ResBlock(channels, norm, bn_first, comm)
                setattr(self, 'block{}'.format(i), layer)

    def forward(self, x):
        h = self.d(x)
        for i in range(self.n_blocks):
            h = getattr(self, 'block{}'.format(i))(h)
        return h


class VD(chainer.Chain):
    """VAE (down to 256dim.) see Table1"""
    def __init__(
            self,
            channels,
            norm=GroupNormalization,
            bn_first=True,
            ndim_latent=128,
            comm=None
    ):
        super(VD, self).__init__()
        with self.init_scope():
            if bn_first:
                self.b = NRC(channels, 16, norm, True, comm)
            else:
                self.b = CNR(16, norm, True, comm)
            self.d_mu = L.Linear(None, ndim_latent)
            self.d_ln_var = L.Linear(None, ndim_latent)

    def forward(self, x):
        h = self.b(x)
        mu = self.d_mu(h)
        ln_var = self.d_ln_var(h)
        return mu, ln_var


class Encoder(chainer.Chain):

    def __init__(
            self,
            base_channels=32,
            norm=GroupNormalization,
            bn_first=True,
            ndim_latent=128,
            comm=None
    ):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.enc_initconv = L.Convolution3D(None, base_channels, 3, 1, 1)
            self.enc_block0 = DownBlock(base_channels, norm, False, 1, bn_first, comm)
            self.enc_block1 = DownBlock(2*base_channels, norm, True, 2, bn_first, comm)
            self.enc_block2 = DownBlock(4*base_channels, norm, True, 2, bn_first, comm)
            self.enc_block3 = DownBlock(8*base_channels, norm, True, 4, bn_first, comm)

    def forward(self, x):
        h = F.dropout(self.enc_initconv(x), ratio=0.2)
        hs = []
        for i in range(4):
            h = getattr(self, 'enc_block{}'.format(i))(h)
            hs.append(h)
        return hs


class UpBlock(chainer.Chain):

    def __init__(
            self,
            channels,
            norm=GroupNormalization,
            bn_first=True,
            mode='sum',
            comm=None
    ):
        self.mode = mode
        concat_mode = True if mode == 'concat' else False
        super(UpBlock, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution3D(None, channels, 1, 1, 0)
            self.rb = ResBlock(channels, norm, bn_first, comm, concat_mode)

    def forward(self, xd, xu=None):
        xd_shape = xd.shape[2:]
        out_shape = tuple(i * 2 for i in xd_shape)
        h = resize_images_3d(self.c1(xd), output_shape=out_shape)
        if xu is not None:
            if self.mode == 'sum':
                h += xu
            elif self.mode == 'concat':
                h = F.concat((h, xu), axis=1)
        h = self.rb(h)
        return h


class Decoder(chainer.Chain):

    def __init__(
            self,
            base_channels,
            out_channels,
            norm=GroupNormalization,
            bn_first=True,
            mode='sum',
            comm=None
    ):
        super(Decoder, self).__init__()
        with self.init_scope():
            for i in range(3):
                layer = UpBlock(2**i * base_channels, norm, bn_first, mode, comm)
                setattr(self, 'dec_block{}'.format(i), layer)
            self.dec_end = L.Convolution3D(None, out_channels, 1, 1, 0)

    def forward(self, hs, bs=None):
        # bs: output from boundary stream
        y = hs[-1]
        for i in reversed(range(3)):
            h = hs[i]
            y = getattr(self, 'dec_block{}'.format(i))(y, h)
        if bs is not None:
            y = F.concat((y, bs), axis=1)
        y = self.dec_end(y)
        return y


class VU(chainer.Chain):

    def __init__(
            self,
            input_shape
    ):
        self.bottom_shape = tuple(i // 16 for i in input_shape)
        bottom_size = 16 * np.prod(list(self.bottom_shape))
        self.output_shape = tuple(i // 8 for i in input_shape)
        super(VU, self).__init__()
        with self.init_scope():
            self.dense = L.Linear(None, bottom_size)
            self.conv1 = L.Convolution3D(None, 256, 1, 1, 0)

    def forward(self, x):
        x_shape = x.shape
        h = F.relu(self.dense(x))
        h = F.reshape(h, (x_shape[0], 16) + self.bottom_shape)
        h = resize_images_3d(self.conv1(h), output_shape=self.output_shape)
        return h


class VAE(chainer.Chain):

    def __init__(
            self,
            in_channels,
            base_channels,
            norm=GroupNormalization,
            bn_first=True,
            input_shape=(160, 192, 128),
            comm=None
    ):
        super(VAE, self).__init__()
        with self.init_scope():
            self.vu = VU(input_shape)
            for i in range(3):
                layer = UpBlock(2**i * base_channels, norm, bn_first, None, comm)
                setattr(self, 'vae_block{}'.format(i), layer)
            self.vae_end = L.Convolution3D(None, in_channels, 1, 1, 0)

    def forward(self, x):
        h = self.vu(x)
        for i in reversed(range(3)):
            h = getattr(self, 'vae_block{}'.format(i))(h)
        h = self.vae_end(h)
        return h


def divide_img(x, grid_size=32):
    b, c, x1, x2, x3 = x.shape
    kersize = grid_size  # kernel size
    ssize = int(grid_size*0.5)  # stride size
    gl0 = int(x1/ssize-1)  # grid length
    gl1 = int(x2/ssize-1)
    gl2 = int(x3/ssize-1)

    if type(x) == chainer.variable.Variable:
        h = im2col_nd(
            x.data, ksize=(kersize, kersize, kersize),
            stride=(ssize, ssize, ssize), pad=(0, 0, 0))
    else:
        h = im2col_nd(
            x, ksize=(kersize, kersize, kersize),
            stride=(ssize, ssize, ssize), pad=(0, 0, 0))
    h = F.reshape(h, (1, 4, kersize, kersize, kersize, gl0*gl1*gl2))
    h = F.transpose(h, axes=(5, 1, 2, 3, 4, 0))
    h = F.reshape(h, (gl0*gl1*gl2, 4, kersize, kersize, kersize))
    return h


class CPCPredictor(chainer.Chain):

    def __init__(
            self,
            base_channels=256,
            norm=GroupNormalization,
            bn_first=True,
            grid_size=32,
            input_shape=(160, 160, 128),
            upper=True,  # whether to predict the upper half in the CPC task
            cpc_pattern='updown',
            comm=None
    ):
        x1, x2, x3 = input_shape
        ssize = int(grid_size*0.5)
        self.gl0 = int(x1/ssize-1)
        self.gl1 = int(x2/ssize-1)
        self.gl2 = int(x3/ssize-1)
        self.cut_l = int(self.gl2/2)
        self.base_channels = base_channels
        self.upper = upper
        self.cpc_pattern = cpc_pattern

        super(CPCPredictor, self).__init__()
        with self.init_scope():
            for i in range(8):
                layer = ResBlock(base_channels, norm, bn_first, comm)
                setattr(self, 'pred_block{}'.format(i), layer)
            self.pred1 = L.Convolution3D(
                None, base_channels,
                ksize=(1, 1, 1), stride=1, pad=0)

    def forward(self, x):
        h = F.transpose(x, axes=(1, 0))
        h = F.reshape(h, (1, self.base_channels, self.gl0, self.gl1, self.gl2))
        if self.cpc_pattern == 'ichimatsu':
            hs = h[:, :, 0:self.gl0:4, 0:self.gl1:4, 0:6:2]
        else:
            if self.upper:
                hs = h[:, :, :, :, :self.cut_l]
            else:
                hs = h[:, :, :, :, -self.cut_l:]
        for i in range(8):
            hs = getattr(self, 'pred_block{}'.format(i))(hs)
        hs = self.pred1(hs)
        return hs


class Attention(chainer.Chain):
    """concatenate inputs from the attention boundary stream,
    and from the Encoder. Then apply a 1*1 conv and a sigmoid activation."""
    def __init__(
            self,
            comm=None
    ):
        super(Attention, self).__init__()
        with self.init_scope():
            self.c = L.Convolution3D(None, out_channels=1, ksize=1, stride=1, pad=0)

    def forward(self, s, m):
        alpha = F.concat((s, m), axis=1)
        alpha = F.sigmoid(self.c(alpha))
        o = s*alpha
        return o


class BoundaryStream(chainer.Chain):
    """Combination of upblock and attention for the boundary stream"""
    def __init__(
            self,
            base_channels,
            out_channels,
            norm=GroupNormalization,
            bn_first=True,
            mode='sum',
            comm=None
    ):
        super(BoundaryStream, self).__init__()
        with self.init_scope():
            self.c = L.Convolution3D(None, 2**2*base_channels, 1, 1, 0)
            self.rb = ResBlock(2**2*base_channels, norm, bn_first, comm)
            self.att = Attention(comm)
            for i in range(3):
                layer = UpBlock(2**i * base_channels, norm, bn_first, mode, comm)
                setattr(self, 'dec_block{}'.format(i), layer)
                att = Attention(comm)
                setattr(self, 'att_block{}'.format(i), att)
            self.dec_end = L.Convolution3D(None, out_channels, ksize=1, stride=1, pad=0)

    def forward(self, hs):
        bs = hs[-1]
        bs = self.att(self.rb(self.c(bs)), hs[-1])
        for i in reversed(range(3)):
            h = hs[i]
            bs = getattr(self, 'dec_block{}'.format(i))(bs)
            bs = getattr(self, 'att_block{}'.format(i))(bs, h)
        y = self.dec_end(bs)
        return y, bs
