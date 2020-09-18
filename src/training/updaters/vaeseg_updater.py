import numpy as np
import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer import reporter
from chainer import Variable
from chainer.training import StandardUpdater
from src.functions.evaluation import dice_coefficient, mean_dice_coefficient
from src.functions.loss.mixed_dice_loss import dice_loss_plus_cross_entropy


class VAESegUpdater(StandardUpdater):

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
        self.enc_freeze = config['enc_freeze']
        self.idle_weight = config['vae_idle_weight']
        super(VAESegUpdater, self).__init__(**kwargs)

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
        opt_em, emb = self.get_optimizer_and_model('emb')
        opt_d, dec = self.get_optimizer_and_model('dec')
        opt_v, vae = self.get_optimizer_and_model('vae')

        if self.is_new_epoch:
            decay_rate = (1. - float(self.epoch / self.nb_epoch)) ** 0.9
            if self.optimizer_name == 'Adam':
                if self.init_weight is not None:
                    opt_e.alpha = self.init_lr*self.enc_freeze * decay_rate
                else:
                    opt_e.alpha = self.init_lr * decay_rate
                opt_em.alpha = self.init_lr * decay_rate
                opt_d.alpha = self.init_lr * decay_rate
                opt_v.alpha = self.init_lr * decay_rate
            else:
                if self.init_weight is not None:
                    opt_e.lr = self.init_lr*self.enc_freeze * decay_rate
                else:
                    opt_e.lr = self.init_lr * decay_rate
                opt_em.lr = self.init_lr * decay_rate
                opt_d.lr = self.init_lr * decay_rate
                opt_v.lr = self.init_lr * decay_rate

        x, t, batchsize = self.get_batch()

        hs = enc(x)
        mu, ln_var = emb(hs[-1])
        latent_size = np.prod(list(mu.shape))

        kl_loss = F.gaussian_kl_divergence(mu, ln_var, reduce='sum') / latent_size

        rec_loss = 0.
        for i in range(self.k):
            z = F.gaussian(mu, ln_var)
            rec_x = vae(z)
            rec_loss += F.mean_squared_error(x, rec_x)
        rec_loss /= self.k

        opt_e.target.cleargrads()
        opt_em.target.cleargrads()
        opt_v.target.cleargrads()
        opt_d.target.cleargrads()
        xp = cuda.get_array_module(t)
        if xp.sum(t) < -(2**31):
            # if label is NaN, optimize encoder, VAE only
            weighted_loss = self.idle_weight*(
                self.rec_loss_weight
                * rec_loss + self.kl_loss_weight * kl_loss)
            t = xp.zeros(t.shape, dtype=np.int32)
            y = dec(hs)

        else:
            y = dec(hs)
            if self.nested_label:
                seg_loss = 0.
                for i in range(t.shape[1]):
                    seg_loss += self.seg_lossfun(y[:, 2*i:2*(i+1), ...], t[:, i, ...])
            else:
                seg_loss = self.seg_lossfun(y, t)
            if self.pretrain:
                weighted_loss = self.rec_loss_weight * rec_loss + self.kl_loss_weight * kl_loss
            else:
                weighted_loss = seg_loss + self.rec_loss_weight * rec_loss \
                    + self.kl_loss_weight * kl_loss

        weighted_loss.backward()
        opt_e.update()
        opt_em.update()
        opt_v.update()
        opt_d.update()

        self.report_scores(y, t)
        reporter.report({
            'loss/rec': rec_loss,
            'loss/kl': kl_loss,
            'loss/total': weighted_loss
        })
