import numpy as np
import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer import reporter
from chainer import Variable
from chainer.training import StandardUpdater
from src.functions.loss.mixed_dice_loss import dice_loss_plus_cross_entropy
from src.functions.evaluation import dice_coefficient, mean_dice_coefficient
from src.functions.loss.cpc_loss import cpc_loss
from src.links.model.vaeseg import divide_img


class CPCSegUpdater(StandardUpdater):

    def __init__(self, config, **kwargs):
        self.nested_label = config['nested_label']
        self.seg_lossfun = eval(config['seg_lossfun'])
        self.k = config['vaeseg_nb_sampling']
        self.rec_loss_weight = config['vaeseg_rec_loss_weight']
        self.kl_loss_weight = config['vaeseg_kl_loss_weight']
        self.optimizer_name = config['optimizer']
        self.init_lr = config['init_lr']
        self.nb_epoch = config['epoch']
        self.pretrain = config['pretrain']
        self.init_weight = config['init_encoder']
        self.grid_size = config['grid_size']
        self.base_channels = config['base_channels']
        self.cpc_loss_weight = config['cpc_vaeseg_cpc_loss_weight']
        self.enc_freeze = config['enc_freeze']
        self.cpc_pattern = config['cpc_pattern']
        self.idle_weight = config['vae_idle_weight']
        super(CPCSegUpdater, self).__init__(**kwargs)

    def get_optimizer_and_model(self, key):
        optimizer = self.get_optimizer(key)
        return optimizer, optimizer.target

    def get_batch(self):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_arrays = self.converter(batch, self.device)
        return Variable(in_arrays[0]), in_arrays[1], batchsize

    def report_scores(self, y, t):
        with chainer.no_backprop_mode():
            if self.nested_label:
                dice = mean_dice_coefficient(dice_coefficient(y[:, 0:2, ...], t[:, 0, ...]))
                for i in range(1, t.shape[1]):
                    dices = dice_coefficient(y[:, 2 * i:2 * (i + 1), ...], t[:, i, ...])
                    dice = F.concat((dice, mean_dice_coefficient(dices)), axis=0)
            else:
                dice = dice_coefficient(y, t)
            mean_dice = mean_dice_coefficient(dice)
            if self.nested_label:
                b, c, h, w, d = t.shape
                y = F.reshape(y, (b, 2, h * c, w, d))
                t = F.reshape(t, (b, h * c, w, d))
            accuracy = F.accuracy(y, t)

        reporter.report({
            'acc': accuracy,
            'mean_dc': mean_dice
        })
        xp = cuda.get_array_module(y)
        for i in range(len(dice)):
            if not xp.isnan(dice.data[i]):
                reporter.report({'dc_{}'.format(i): dice[i]})

    def update_core(self):
        opt_e, enc = self.get_optimizer_and_model('enc')
        opt_d, dec = self.get_optimizer_and_model('dec')

        opt_p1, cpcpred1 = self.get_optimizer_and_model('cpcpred1')

        if self.is_new_epoch:
            decay_rate = (1. - float(self.epoch / self.nb_epoch)) ** 0.9
            if self.optimizer_name == 'Adam':
                if self.init_weight is not None:
                    opt_e.alpha = self.init_lr*self.enc_freeze * decay_rate
                else:
                    opt_e.alpha = self.init_lr * decay_rate
                opt_d.alpha = self.init_lr * decay_rate
                opt_p1.alpha = self.init_lr * decay_rate
            else:
                if self.init_weight is not None:
                    opt_e.lr = self.init_lr*self.enc_freeze * decay_rate
                else:
                    opt_e.lr = self.init_lr * decay_rate
                opt_d.lr = self.init_lr * decay_rate
                opt_p1.lr = self.init_lr * decay_rate

        x, t, batchsize = self.get_batch()

        b, c, x1, x2, x3 = x.shape
        ssize = int(self.grid_size*0.5)  # stride size
        gl0 = int(x1/ssize-1)  # grid length
        gl1 = int(x2/ssize-1)
        gl2 = int(x3/ssize-1)

        hs = enc(x)
        h2 = divide_img(x)
        h2 = enc(h2)
        h2 = F.average(h2[-1], axis=(2, 3, 4))
        cpc_t = F.transpose(h2, axes=(1, 0))
        cpc_t = F.reshape(cpc_t, (1, self.base_channels*8, gl0, gl1, gl2))

        y1 = cpcpred1(h2)
        cpc_loss1 = cpc_loss(y1, cpc_t, upper=True, cpc_pattern=self.cpc_pattern)
        cpc_loss_tot = cpc_loss1

        y = dec(hs)

        opt_e.target.cleargrads()
        opt_d.target.cleargrads()
        opt_p1.target.cleargrads()

        xp = cuda.get_array_module(t)
        if xp.sum(t) < -(2**31):
            weighted_loss = self.idle_weight * self.cpc_loss_weight * cpc_loss_tot
            weighted_loss.backward()
            t = xp.zeros(t.shape, dtype=np.int32)

        else:
            if self.nested_label:
                seg_loss = 0.
                for i in range(t.shape[1]):
                    seg_loss += self.seg_lossfun(y[:, 2*i:2*(i+1), ...], t[:, i, ...])
            else:
                seg_loss = self.seg_lossfun(y, t)

            weighted_loss = seg_loss + self.cpc_loss_weight*cpc_loss_tot
            weighted_loss.backward()

        opt_e.update()
        opt_d.update()
        opt_p1.update()

        self.report_scores(y, t)
        reporter.report({
            'loss/cpc': cpc_loss_tot,
            'loss/total': weighted_loss
        })
