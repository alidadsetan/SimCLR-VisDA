from pathlib import Path

from dataset.visda_dataset import VisdaDataset, VisdaTrainDataset, VisdaValidDataset
from transformations import contrast_valid_transforms, contrast_train_transforms
from util import create_samples

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--action", choices=["finetune"])
parser.add_argument("--storage", help="location for dataset")
parser.add_argument("--transform-sample-directory", default=str(Path(".")/"transform_sample"))

args = parser.parse_args()

if args.action == "finetune":
    storage_path = Path(args.storage)
    train_dataset = VisdaTrainDataset((storage_path/'train').resolve(), transform=contrast_train_transforms,n_views=5)

    valid_dataset = VisdaValidDataset((storage_path/'validation').resolve(), transform=contrast_valid_transforms)
    valid_dataset.set_param(5)

    unsupervised_dataset = VisdaDataset(train_dataset,valid_dataset)


    create_samples(unsupervised_dataset, Path(args.transform_sample_directory))