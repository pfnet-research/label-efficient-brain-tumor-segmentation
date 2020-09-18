import sys
sys.path.append('.')  # NOQA
from src.datasets.preprocess import split_kfold


def main(root_path, arr_type='nii.gz', n_splits=5, random_state=42):
    # create split file
    split_kfold(root_path, arr_type, n_splits, random_state)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
