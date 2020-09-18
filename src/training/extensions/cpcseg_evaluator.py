import copy
import cupy
import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.training.extensions import Evaluator
from chainer.dataset import convert
from chainer import Reporter
from chainer import reporter as reporter_module
from src.functions.evaluation import dice_coefficient, mean_dice_coefficient
from src.functions.loss.mixed_dice_loss import dice_loss_plus_cross_entropy


class CPCSegEvaluator(Evaluator):

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
        self.rec_loss_weight = config['vaeseg_rec_loss_weight']
        self.kl_loss_weight = config['vaeseg_kl_loss_weight']
        self.grid_size = config['grid_size']
        self.base_channels = config['base_channels']
        self.cpc_loss_weight = config['cpc_vaeseg_cpc_loss_weight']
        self.cpc_pattern = config['cpc_pattern']
        self.is_brats = config['is_brats']
        self.dataset = config['dataset_name']
        self.nb_labels = config['nb_labels']
        self.crop_size = eval(config['crop_size'])

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
                    # evaluation method for BRATS dataset only
                    h, w, d = x.shape[2:]
                    hc, wc, dc = self.crop_size
                    if self.nested_label:
                        y = cupy.zeros((1, 2*(self.nb_labels-1), h, w, d), dtype='float32')
                    else:
                        y = cupy.zeros((1, self.nb_labels, h, w, d), dtype='float32')
                    hker = hc  # kernel size
                    hs = int(0.5*hker)  # stride
                    wker = wc
                    wc = int(0.5*wker)
                    dker = dc  # kernel size for depth
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                xx = x[:, :, -i*hker:min(hker*(i+1), h),
                                       -j*wker:min(wker*(j+1), w), -k*dker:min(dker*(k+1), d)]
                                hs = enc(xx)
                                yy = dec(hs)
                                y[:, :, -i*hker:min(hker*(i+1), h),
                                    -j*wker:min(wker*(j+1), w),
                                    -k*dker:min(dker*(k+1), d)] += yy.data

                else:
                    hs = enc(x)
                    y = dec(hs)
                seg_loss = self.compute_loss(y, t)
                accuracy = self.compute_accuracy(y, t)
                dice = self.compute_dice_coef(y, t)
                mean_dice = mean_dice_coefficient(dice)

                observation = {}
                with reporter.scope(observation):
                    reporter.report({
                        'loss/seg': seg_loss,
                        'acc': accuracy,
                        'mean_dc': mean_dice
                    }, observer)
                    xp = cuda.get_array_module(y)
                    for i in range(len(dice)):
                        if not xp.isnan(dice.data[i]):
                            reporter.report({'dc_{}'.format(i): dice[i]}, observer)
            summary.add(observation)
        return summary.compute_mean()
