import sys
sys.path.append('.')  # NOQA
import argparse
import yaml
import warnings
import numpy as np
import random
import chainer
from chainer import global_config

from src.utils.config import overwrite_config
from src.utils.setup_helpers import setup_trainer

warnings.simplefilter(action='ignore', category=FutureWarning)  # NOQA


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


def main():
    reset_seed(42)
    parser = argparse.ArgumentParser(
        description='Medical image segmentation'
    )
    parser.add_argument('--config', '-c')
    parser.add_argument('--out', default='results',
                        help='Output directory')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help="Number of epochs")
    parser.add_argument('--gpu_start_id', '-g', type=int, default=0,
                        help="Start ID of gpu. (negative value indicates cpu)")
    args = parser.parse_args()
    config = overwrite_config(
        yaml.load(open(args.config)), dump_yaml_dir=args.out
    )

    if config['mn']:
        global_config.autotune = True
        import multiprocessing
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process(target=print, args=('Initialize forkserver',))
        p.start()
        p.join()

    trainer = setup_trainer(config, args.out, args.batch_size, args.epoch, args.gpu_start_id)
    trainer.run()


if __name__ == '__main__':
    main()
