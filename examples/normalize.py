import sys
sys.path.append('.')  # NOQA
from src.datasets.preprocess import normalize


def main(root_path=None, arr_type='nii.gz', modality='mri'):
    # save normalized npz arrays in root_path/normalized/
    normalize(root_path, arr_type, modality)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
