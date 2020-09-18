import copy
import os
import time
import yaml


default_config = {
    'dataset_name': 'msd_bound',
    'image_path': '/PATH_TO_NORMALIZED_IMAGES/imagesTr_normalized',
    'label_path': '/PATH_TO_LABELS/labelsTr',
    'image_file_format': 'npz',
    'label_file_format': 'nii',
    'train_list_path': '/PATH_TO_TRAINING_LIST/train_list_cv0.txt',
    'validation_list_path': '/PATH_TO_VALIDATION_LIST/validation_list_cv0.txt',
    'test_list_path': '/PATH_TO_TEST_LIST/test_list_cv0.txt',
    'target_label': 0,
    'nested_label': False,
    'crop_size': '(160,192,128)',
    'shift_intensity': 0.1,
    'random_flip': True,

    'segmentor_name': 'vaeseg',
    'in_channels': 4,
    'base_channels': 32,
    'nb_labels': 4,
    'vaeseg_norm': 'GroupNormalization',
    'vaeseg_bn_first': True,
    'vaeseg_ndim_latent': 128,
    'vaeseg_skip_connect_mode': 'sum',
    'unet_cropping': False,
    'dv_num_trans_layers': 12,

    'seg_lossfun': 'softmax_dice_loss',
    'auxiliary_weights': None,
    'vaeseg_nb_sampling': 1,
    'vaeseg_rec_loss_weight': 0.1,
    'vaeseg_kl_loss_weight': 0.1,
    'structseg_nb_copies': 1,

    'mn': True,
    'gpu_start_id': 0,
    'loaderjob': 2,
    'batchsize': 1,
    'val_batchsize': 2,
    'epoch': 200,
    'optimizer': 'Adam',
    'init_lr': 1e-4,
    'lr_reduction_ratio': 0.99,
    'lr_reduction_interval': (1, 'epoch'),
    'weight_decay': 1e-5,
    'report_interval': 10,
    'eval_interval': 2,
    'snapshot_interval': 20,
    'init_segmentor': None,
    'init_encoder': None,
    'init_decoder': None,
    'init_vae': None,
    'resume': None,

    'pretrain': False,
    'grid_size': 32,
    'init_embedder': None,
    'cpc_vaeseg_cpc_loss_weight': 0.001,
    'init_cpcpred': None,
    'enc_freeze': 0.01,
    'cpc_pattern': 'updown',
    'random_scale': False,
    'edge_path': '/PATH_TO_EDGE_LABELS/edgesTr',
    'edge_file_format': 'npy',
    'edge_label': False,
    'print_each_dc': True,
    'is_brats': False,
    'ignore_path': None,
    'vae_idle_weight': 1
}


def overwrite_config(
        input_cfg,
        dump_yaml_dir=None
):
    output_cfg = copy.copy(default_config)
    for key, val in input_cfg.items():
        if key not in output_cfg:
            raise ValueError('Unknown configuration key: {}'.format(key))
        output_cfg[key] = val
    if dump_yaml_dir is not None:
        os.makedirs(dump_yaml_dir, exist_ok=True)
        cur_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
        dump_yaml_path = os.path.join(
            dump_yaml_dir, '{}.yaml'.format(cur_time))
        with open(dump_yaml_path, 'w') as f:
            yaml.dump(output_cfg, f)
    return output_cfg
