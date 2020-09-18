from chainer.links import BatchNormalization, GroupNormalization
from chainermn.links import MultiNodeBatchNormalization
from chainer.functions import softmax_cross_entropy
from chainer.optimizers import Adam
from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.backends.cuda import get_device_from_id
import chainermn
from src.datasets.msd_bound import MSDBoundDataset
from src.links.model.vaeseg import BoundaryStream, CPCPredictor, Decoder, Encoder, VAE, VD
from src.training.updaters.vaeseg_updater import VAESegUpdater
from src.training.extensions.vaeseg_evaluator import VAESegEvaluator
from src.training.updaters.encdec_seg_updater import EncDecSegUpdater
from src.training.extensions.encdec_seg_evaluator import EncDecSegEvaluator
from src.training.updaters.boundseg_updater import BoundSegUpdater
from src.training.extensions.boundseg_evaluator import BoundSegEvaluator
from src.training.updaters.cpcseg_updater import CPCSegUpdater
from src.training.extensions.cpcseg_evaluator import CPCSegEvaluator


def _setup_communicator(config, gpu_start_id=0):
    if config['mn']:
        comm = chainermn.create_communicator('pure_nccl')
        is_master = (comm.rank == 0)
        device = comm.intra_rank + gpu_start_id
    else:
        comm = None
        is_master = True
        device = gpu_start_id
    return comm, is_master, device


def _setup_datasets(config, comm, is_master):
    if is_master:
        if config['dataset_name'] == 'msd_bound':
            train_data = MSDBoundDataset(config, config['train_list_path'])
            validation_data = MSDBoundDataset(config, config['validation_list_path'])
            test_data = MSDBoundDataset(config, config['test_list_path'])
            validation_data.random_scale = False
            test_data.random_scale = False
            validation_data.shift_intensity = 0
            test_data.shift_intensity = 0
            validation_data.random_flip = False
            test_data.random_flip = False
            validation_data.nb_copies = 1
            test_data.nb_copies = 1
            validation_data.training = False
            test_data.training = False
        else:
            raise ValueError('Unknown dataset_name: {}'.format(config['dataset_name']))
        print('Training dataset size: {}'.format(len(train_data)))
        print('Validation dataset size: {}'.format(len(validation_data)))
        print('Test dataset size: {}'.format(len(test_data)))
    else:
        train_data = None
        validation_data = None
        test_data = None

    # scatter dataset
    if comm is not None:
        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
        validation_data = chainermn.scatter_dataset(validation_data, comm, shuffle=True)
        test_data = chainermn.scatter_dataset(test_data, comm, shuffle=True)

    return train_data, validation_data, test_data


def _setup_vae_segmentor(config, comm=None):
    in_channels = config['in_channels']
    base_channels = config['base_channels']
    out_channels = config['nb_labels']
    nested_label = config['nested_label']
    norm = eval(config['vaeseg_norm'])
    bn_first = config['vaeseg_bn_first']
    ndim_latent = config['vaeseg_ndim_latent']
    mode = config['vaeseg_skip_connect_mode']
    input_shape = eval(config['crop_size'])
    if nested_label:
        out_channels = 2 * (out_channels - 1)

    encoder = Encoder(
        base_channels=base_channels,
        norm=norm,
        bn_first=bn_first,
        ndim_latent=ndim_latent,
        comm=comm
    )

    embedder = VD(
        channels=8*base_channels,
        norm=norm,
        bn_first=bn_first,
        ndim_latent=ndim_latent,
        comm=comm
    )

    decoder = Decoder(
        base_channels=base_channels,
        out_channels=out_channels,
        norm=norm,
        bn_first=bn_first,
        mode=mode,
        comm=comm
    )

    vae = VAE(
        in_channels=in_channels,
        base_channels=base_channels,
        norm=norm,
        bn_first=bn_first,
        input_shape=input_shape,
        comm=comm
    )

    return encoder, embedder, decoder, vae


def _setup_vae_segmentor_only(config, comm=None):
    base_channels = config['base_channels']
    out_channels = config['nb_labels']
    nested_label = config['nested_label']
    norm = eval(config['vaeseg_norm'])
    bn_first = config['vaeseg_bn_first']
    ndim_latent = config['vaeseg_ndim_latent']
    mode = config['vaeseg_skip_connect_mode']
    if nested_label:
        out_channels = 2 * (out_channels - 1)

    encoder = Encoder(
        base_channels=base_channels,
        norm=norm,
        bn_first=bn_first,
        ndim_latent=ndim_latent,
        comm=comm
    )

    decoder = Decoder(
        base_channels=base_channels,
        out_channels=out_channels,
        norm=norm,
        bn_first=bn_first,
        mode=mode,
        comm=comm
    )

    return encoder, decoder


def _setup_cpc_segmentor(config, comm=None):
    base_channels = config['base_channels']
    out_channels = config['nb_labels']
    nested_label = config['nested_label']
    norm = eval(config['vaeseg_norm'])
    bn_first = config['vaeseg_bn_first']
    ndim_latent = config['vaeseg_ndim_latent']
    mode = config['vaeseg_skip_connect_mode']
    input_shape = eval(config['crop_size'])
    grid_size = config['grid_size']
    cpc_pattern = config['cpc_pattern']
    if nested_label:
        out_channels = 2 * (out_channels - 1)

    encoder = Encoder(
        base_channels=base_channels,
        norm=norm,
        bn_first=bn_first,
        ndim_latent=ndim_latent,
        comm=comm
    )

    decoder = Decoder(
        base_channels=base_channels,
        out_channels=out_channels,
        norm=norm,
        bn_first=bn_first,
        mode=mode,
        comm=comm
    )

    cpcpred1 = CPCPredictor(
        base_channels=base_channels*8,
        norm=norm,
        bn_first=bn_first,
        grid_size=grid_size,
        input_shape=input_shape,
        upper=True,
        cpc_pattern=cpc_pattern,
        comm=comm
    )
    return encoder, decoder, cpcpred1


def _setup_bound_segmentor(config, comm=None):
    base_channels = config['base_channels']
    out_channels = config['nb_labels']
    nested_label = config['nested_label']
    norm = eval(config['vaeseg_norm'])
    bn_first = config['vaeseg_bn_first']
    mode = config['vaeseg_skip_connect_mode']
    ndim_latent = config['vaeseg_ndim_latent']
    if nested_label:
        out_channels = 2 * (out_channels - 1)

    encoder = Encoder(
        base_channels=base_channels,
        norm=norm,
        bn_first=bn_first,
        ndim_latent=ndim_latent,
        comm=comm
    )

    decoder = Decoder(
        base_channels=base_channels,
        out_channels=out_channels,
        norm=norm,
        bn_first=bn_first,
        mode=mode,
        comm=comm
    )

    boundary = BoundaryStream(
        base_channels=base_channels,
        out_channels=out_channels,
        norm=norm,
        comm=comm
    )

    return encoder, decoder, boundary


def _setup_iterators(config, batch_size, train_data, validation_data, test_data):
    if isinstance(config['loaderjob'], int) and config['loaderjob'] > 1:
        train_iterator = MultiprocessIterator(
            train_data, batch_size, n_processes=config['loaderjob'])
        validation_iterator = MultiprocessIterator(
            validation_data, batch_size, n_processes=config['loaderjob'],
            repeat=False, shuffle=False)
        test_iterator = MultiprocessIterator(
            test_data, batch_size, n_processes=config['loaderjob'],
            repeat=False, shuffle=False)
    else:
        train_iterator = SerialIterator(train_data, batch_size)
        validation_iterator = SerialIterator(
            validation_data, batch_size, repeat=False, shuffle=False)
        test_iterator = SerialIterator(
            test_data, batch_size, repeat=False, shuffle=False)

    return train_iterator, validation_iterator, test_iterator


# Optimizer
def _setup_optimizer(config, model, comm):
    optimizer_name = config['optimizer']
    lr = float(config['init_lr'])
    weight_decay = float(config['weight_decay'])
    if optimizer_name == 'Adam':
        optimizer = Adam(alpha=lr, weight_decay_rate=weight_decay)
    elif optimizer_name in \
            ('SGD', 'MomentumSGD', 'CorrectedMomentumSGD', 'RMSprop'):
        optimizer = eval(optimizer_name)(lr=lr)
        if weight_decay > 0.:
            optimizer.add_hook(WeightDecay(weight_decay))
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer_name))
    if comm is not None:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)

    return optimizer


# Updater
def _setup_updater(config, device, train_iterator, optimizers):
    updater_kwargs = dict()
    updater_kwargs['iterator'] = train_iterator
    updater_kwargs['optimizer'] = optimizers
    updater_kwargs['device'] = device

    if config['segmentor_name'] == 'vaeseg':
        return VAESegUpdater(config, **updater_kwargs)
    elif config['segmentor_name'] == 'encdec_seg':
        return EncDecSegUpdater(config, **updater_kwargs)
    elif config['segmentor_name'] == 'boundseg':
        return BoundSegUpdater(config, **updater_kwargs)
    elif config['segmentor_name'] == 'cpcseg':
        return CPCSegUpdater(config, **updater_kwargs)
    else:
        return training.StandardUpdater(**updater_kwargs)


def _setup_extensions(config, trainer, optimizers, logging_counts, logging_attributes):
    if config['segmentor_name'] == 'vaeseg':
        trainer.extend(extensions.dump_graph('loss/total', out_name="segmentor.dot"))
    elif config['segmentor_name'] == 'encdec_seg':
        trainer.extend(extensions.dump_graph('loss/seg', out_name="segmentor.dot"))
    elif config['segmentor_name'] == 'boundseg':
        trainer.extend(extensions.dump_graph('loss/seg', out_name="segmentor.dot"))
    elif config['segmentor_name'] == 'cpcseg':
        trainer.extend(extensions.dump_graph('loss/total', out_name="segmentor.dot"))
    else:
        trainer.extend(extensions.dump_graph('main/loss', out_name="segmentor.dot"))

    # Report
    repo_trigger = (config['report_interval'], 'iteration')
    trainer.extend(
        extensions.LogReport(
            trigger=repo_trigger
        )
    )
    trainer.extend(
        extensions.PrintReport(logging_counts + logging_attributes),
        trigger=repo_trigger
    )
    trainer.extend(
        extensions.ProgressBar()
    )

    snap_trigger = (config['snapshot_interval'], 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'),
        trigger=snap_trigger
    )
    for k, v in optimizers.items():
        trainer.extend(
            extensions.snapshot_object(v.target, k+'_epoch_{.updater.epoch}'),
            trigger=snap_trigger
        )

    for attr in logging_attributes:
        trainer.extend(
            extensions.PlotReport([attr, 'validation/' + attr], 'epoch',
                                  file_name=attr.replace('/', '_') + '.png')
        )


# Trainer
def setup_trainer(config, out, batch_size, epoch, gpu_start_id):

    comm, is_master, device = _setup_communicator(config, gpu_start_id)

    train_data, validation_data, test_data = _setup_datasets(config, comm, is_master)

    if config['segmentor_name'] == 'vaeseg':
        encoder, embedder, decoder, vae = _setup_vae_segmentor(config, comm)
        # load weights
        if config['init_encoder'] is not None:
            serializers.load_npz(config['init_encoder'], encoder)
        if config['init_embedder'] is not None:
            serializers.load_npz(config['init_embedder'], embedder)
        if config['init_decoder'] is not None:
            serializers.load_npz(config['init_decoder'], decoder)
        if config['init_vae'] is not None:
            serializers.load_npz(config['init_vae'], vae)
        if device is not None:
            get_device_from_id(device).use()
            encoder.to_gpu()
            embedder.to_gpu()
            decoder.to_gpu()
            vae.to_gpu()

        opt_enc = _setup_optimizer(config, encoder, comm)
        opt_emb = _setup_optimizer(config, embedder, comm)
        opt_dec = _setup_optimizer(config, decoder, comm)
        opt_vae = _setup_optimizer(config, vae, comm)
        optimizers = {'enc': opt_enc, 'emb': opt_emb, 'dec': opt_dec, 'vae': opt_vae}

    elif config['segmentor_name'] == 'cpcseg':

        encoder, decoder, cpcpred1 = _setup_cpc_segmentor(config, comm)
        # load weights
        if config['init_encoder'] is not None:
            serializers.load_npz(config['init_encoder'], encoder)
        if config['init_decoder'] is not None:
            serializers.load_npz(config['init_decoder'], decoder)
        if config['init_cpcpred'] is not None:
            serializers.load_npz(config['init_cpcpred'], cpcpred1)
        if device is not None:
            get_device_from_id(device).use()
            encoder.to_gpu()
            decoder.to_gpu()
            cpcpred1.to_gpu()

        opt_enc = _setup_optimizer(config, encoder, comm)
        opt_dec = _setup_optimizer(config, decoder, comm)
        opt_p1 = _setup_optimizer(config, cpcpred1, comm)
        optimizers = {'enc': opt_enc, 'dec': opt_dec, 'cpcpred1': opt_p1}

    elif config['segmentor_name'] == 'encdec_seg':

        encoder, decoder = _setup_vae_segmentor_only(config, comm)
        # load weights
        if config['init_encoder'] is not None:
            serializers.load_npz(config['init_encoder'], encoder)
        if config['init_decoder'] is not None:
            serializers.load_npz(config['init_decoder'], decoder)
        if device is not None:
            get_device_from_id(device).use()
            encoder.to_gpu()
            decoder.to_gpu()

        opt_enc = _setup_optimizer(config, encoder, comm)
        opt_dec = _setup_optimizer(config, decoder, comm)
        optimizers = {'enc': opt_enc, 'dec': opt_dec}

    elif config['segmentor_name'] == 'boundseg':

        encoder, decoder, boundary = _setup_bound_segmentor(config, comm)
        # load weights
        if config['init_encoder'] is not None:
            serializers.load_npz(config['init_encoder'], encoder)
        if config['init_decoder'] is not None:
            serializers.load_npz(config['init_decoder'], decoder)
        if device is not None:
            get_device_from_id(device).use()
            encoder.to_gpu()
            decoder.to_gpu()
            boundary.to_gpu()

        opt_enc = _setup_optimizer(config, encoder, comm)
        opt_dec = _setup_optimizer(config, decoder, comm)
        opt_bound = _setup_optimizer(config, boundary, comm)
        optimizers = {'enc': opt_enc, 'dec': opt_dec, 'bound': opt_bound}

    train_iterator, validation_iterator, test_iterator = \
        _setup_iterators(config, batch_size, train_data, validation_data, test_data)

    logging_counts = ['epoch', 'iteration']
    if config['segmentor_name'] == 'vaeseg':
        logging_attributes = \
            ['loss/rec', 'loss/kl', 'loss/total', 'acc',
             'mean_dc', 'val/mean_dc', 'test/mean_dc']
        if config['print_each_dc']:
            for i in range(0, config['nb_labels']):
                logging_attributes.append('dc_{}'.format(i))
                logging_attributes.append('val/dc_{}'.format(i))
                logging_attributes.append('test/dc_{}'.format(i))

    elif config['segmentor_name'] == 'cpcseg':
        logging_attributes = \
            ['loss/total', 'acc', 'loss/cpc']
        for i in range(0, config['nb_labels']):
            logging_attributes.append('dc_{}'.format(i))
            logging_attributes.append('val/dc_{}'.format(i))
            logging_attributes.append('test/dc_{}'.format(i))

    elif config['segmentor_name'] == 'encdec_seg':
        logging_attributes = \
            ['loss/seg', 'loss/total', 'acc']
        if config['print_each_dc']:
            for i in range(0, config['nb_labels']):
                logging_attributes.append('dc_{}'.format(i))
                logging_attributes.append('val/dc_{}'.format(i))
                logging_attributes.append('test/dc_{}'.format(i))

    elif config['segmentor_name'] == 'boundseg':
        logging_attributes = \
            ['loss/seg', 'loss/total', 'acc', 'loss/bound', 'loss/bce']
        if config['print_each_dc']:
            for i in range(0, config['nb_labels']):
                logging_attributes.append('dc_{}'.format(i))
                logging_attributes.append('val/dc_{}'.format(i))
                logging_attributes.append('test/dc_{}'.format(i))

    else:
        logging_attributes = ['main/loss', 'main/acc']
        for i in range(1, config['nb_labels']):
            logging_attributes.append('main/dc_{}'.format(i))
            logging_attributes.append('val/main/dc_{}'.format(i))
            logging_attributes.append('test/main/dc_{}'.format(i))

    updater = _setup_updater(config, device, train_iterator, optimizers)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    if is_master:
        _setup_extensions(config, trainer, optimizers, logging_counts, logging_attributes)

    if config['segmentor_name'] == 'vaeseg':
        targets = {'enc': encoder, 'emb': embedder, 'dec': decoder, 'vae': vae}
        val_evaluator = VAESegEvaluator(config, validation_iterator, targets, device=device)
        test_evaluator = VAESegEvaluator(config, test_iterator, targets, device=device)

    elif config['segmentor_name'] == 'cpcseg':
        targets = {'enc': encoder, 'dec': decoder, 'cpcpred1': cpcpred1}
        val_evaluator = CPCSegEvaluator(config, validation_iterator, targets, device=device)
        test_evaluator = CPCSegEvaluator(config, test_iterator, targets, device=device)

    elif config['segmentor_name'] == 'encdec_seg':
        targets = {'enc': encoder, 'dec': decoder}
        val_evaluator = EncDecSegEvaluator(config, validation_iterator, targets, device=device)
        test_evaluator = EncDecSegEvaluator(config, test_iterator, targets, device=device)

    elif config['segmentor_name'] == 'boundseg':
        targets = {'enc': encoder, 'dec': decoder, 'bound': boundary}
        val_evaluator = BoundSegEvaluator(config, validation_iterator, targets, device=device)
        test_evaluator = BoundSegEvaluator(config, test_iterator, targets, device=device)

    val_evaluator.default_name = 'val'
    test_evaluator.default_name = 'test'

    if comm is not None:
        val_evaluator = chainermn.create_multi_node_evaluator(val_evaluator, comm)
        test_evaluator = chainermn.create_multi_node_evaluator(test_evaluator, comm)
    trainer.extend(val_evaluator, trigger=(config['eval_interval'], 'epoch'))
    trainer.extend(test_evaluator, trigger=(config['eval_interval'], 'epoch'))

    # Resume
    if config['resume'] is not None:
        serializers.load_npz(config['resume'], trainer)

    return trainer
