from dataset import LegacyVisdaDataset
from pathlib import Path
from torchvision import datasets

from dataset.visda_dataset import VisdaDataset, VisdaTrainDataset
from .transformations import contrast_transforms
from .util import create_samples

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["finetune"])
parser.add_argument("--storage", help="location for dataset")
parser.add_argument("--transform-sample-directory", default=str(Path(".")/"transform_sample"))

args = parser.parse_args()

if args.action == "finetune":
    storage_path = Path(args.storage)
    train_dataset = VisdaTrainDataset((storage_path/'train').resolve(), transform=contrast_transforms)
    train_dataset.set_params()

    valid_dataset = datasets.ImageFolder((storage_path/'valid').resolve(), transform=contrast_transforms)

    unsupervised_dataset = VisdaDataset(train_dataset,valid_dataset)


    create_samples(unsupervised_dataset, Path(args.transform_sample_directory))