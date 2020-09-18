import copy
import cupy
import math
import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.training.extensions import Evaluator
from chainer.dataset import convert
from chainer import Reporter
from chainer import reporter as reporter_module
from src.functions.evaluation import dice_coefficient, mean_dice_coefficient
from src.functions.loss.mixed_dice_loss import dice_loss_plus_cross_entropy


class EncDecSegEvaluator(Evaluator):

    def __init__(
            self,
            config,
            iterator,
            target,
            converter=convert.concat_examples,
            device=None,
    ):
        super().__init__(iterator, target,
                         converter, device,
                         None, None)
        self.nested_label = config['nested_label']
        self.seg_lossfun = eval(config['seg_lossfun'])
        self.dataset = config['dataset_name']
        self.nb_labels = config['nb_labels']
        self.crop_size = eval(config['crop_size'])
        self.is_brats = config['is_brats']

    def compute_loss(self, y, t):
        if self.nested_label:
            loss = 0.
            b, c, h, w, d = t.shape
            for i in range(c):
                loss += self.seg_lossfun(y[:, 2*i:2*(i+1), ...], t[:, i, ...])
        else:
            loss = self.seg_lossfun(y, t)
        return loss

    def compute_accuracy(self, y, t):
        if self.nested_label:
            b, c, h, w, d = t.shape
            y = F.reshape(y, (b, 2, h*c, w, d))
            t = F.reshape(t, (b, h*c, w, d))
        return F.accuracy(y, t)

    def compute_dice_coef(self, y, t):
        if self.nested_label:
            dice = mean_dice_coefficient(dice_coefficient(y[:, 0:2, ...], t[:, 0, ...]))
            for i in range(1, t.shape[1]):
                dices = dice_coefficient(y[:, 2*i:2*(i+1), ...], t[:, i, ...])
                dice = F.concat((dice, mean_dice_coefficient(dices)), axis=0)
        else:
            dice = dice_coefficient(y, t, is_brats=self.is_brats)
        return dice

    def evaluate(self):
        summary = reporter_module.DictSummary()
        iterator = self._iterators['main']
        enc = self._targets['enc']
        dec = self._targets['dec']
        reporter = Reporter()
        observer = object()
        reporter.add_observer(self.default_name, observer)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        for batch in it:
            x, t = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                if self.dataset == 'msd_bound':
                    h, w, d = x.shape[2:]
                    hc, wc, dc = self.crop_size
                    if self.nested_label:
                        y = cupy.zeros((1, 2*(self.nb_labels-1), h, w, d), dtype='float32')
                    else:
                        y = cupy.zeros((1, self.nb_labels, h, w, d), dtype='float32')
                    s = 128  # stride
                    ker = 256  # kernel size
                    dker = dc  # kernel size for depth
                    ds = dker*0.5  # stride for depth
                    dsteps = int(math.floor((d-dker)/ds) + 1)
                    steps = round((h - ker)/s + 1)
                    for i in range(steps):
                        for j in range(steps):
                            for k in range(dsteps):
                                xx = x[:, :, s*i:ker+s*i, s*j:ker+s*j, ds*k:dker+ds*k]
                                hs = enc(xx)
                                yy = dec(hs)
                                y[:, :, s*i:ker+s*i, s*j:ker+s*j, ds*k:dker+ds*k] += yy.data
                            # for the bottom depth part of the image
                            xx = x[:, :, s*i:ker+s*i, s*j:ker+s*j, -dker:]
                            hs = enc(xx)
                            yy = dec(hs)
                            y[:, :, s*i:ker+s*i, s*j:ker+s*j, -dker:] += yy.data
                else:
                    hs = enc(x)
                    y = dec(hs)
                seg_loss = self.compute_loss(y, t)
                accuracy = self.compute_accuracy(y, t)
                dice = self.compute_dice_coef(y, t)
                mean_dice = mean_dice_coefficient(dice)
                weighted_loss = seg_loss

                observation = {}
                with reporter.scope(observation):
                    reporter.report({
                        'loss/seg': seg_loss,
                        'loss/total': weighted_loss,
                        'acc': accuracy,
                        'mean_dc': mean_dice
                    }, observer)
                    xp = cuda.get_array_module(y)
                    for i in range(len(dice)):
                        if not xp.isnan(dice.data[i]):
                            reporter.report({'dc_{}'.format(i): dice[i]}, observer)
            summary.add(observation)
        return summary.compute_mean()
