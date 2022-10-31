from pathlib import Path
from SimCLR import SimCLR
from dataset.visda_dataset import VisdaUnsupervisedDataset, VisdaTrainDataset, VisdaValidDataset
from transformations import transform_builder
from util import create_samples
from torch.utils.data import random_split
from evaluation_callback import SSLOnlineEvaluator
from torchvision.datasets.folder import ImageFolder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import argparse, argcomplete
from evaluator_module import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--action", choices=["pretrain", "evaluate"])
parser.add_argument("--storage", help="location for dataset")
parser.add_argument("--transform-sample-directory",
                    default=str(Path(".")/"transform_sample"))
parser.add_argument("--checkpoint-directory",
                    default=str((Path(".")/"checkpoints").resolve()))
parser.add_argument("--finetune-every-n-epoch",
                    type=int, default=1)
parser.add_argument("--image-height", type=int, default=320)
parser.add_argument("--pretrain-epochs", type=int, default=2000)
parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
parser.add_argument("--pretrain-batch-size", type=int, default=256)
parser.add_argument("--pretrain-finetune-precentage", type=int, default=1)
parser.add_argument("--use-all-features", action="store_true", default=False)
parser.add_argument("--model-name", type=str,default='efficientnetv2_rw_s')

parser.add_argument("--log-directory", type=str,
                    default=(Path('.')/"logs").resolve())
parser.add_argument("--finetune-batchsize", type=int, default=256)
parser.add_argument("--finetune-epochs", type=int, default=10)
parser.add_argument("--finetune-percentage", type=int, default=1)
parser.add_argument("--save-top-k-models", type=int, default=10)
parser.add_argument("--save-models-every-n-epoch", type=int, default=1)

parser.add_argument("--mlp-output-dimension", type=int, default=128)
parser.add_argument("--high-penalty-weight", type=float, default=10)
parser.add_argument("--low-penalty-weight", type=float, default=.1)
parser.add_argument("--pretrained-weights-path", type=str,required=False)
parser.add_argument("--evaluator-hidden-dim", type=int,default=0)
parser.add_argument("--num-gpus", type=int,default=1)


parser.add_argument("--evaluation-same-dist-val-percentage", type=int,default=5)
parser.add_argument("--evaluation-train-percentage", type=int,default=95)
parser.add_argument("--drop-p", type=float, default=.5)

argcomplete.autocomplete(parser)
args = parser.parse_args()

if args.action == "pretrain":
    # freeze_support()
    storage_path = Path(args.storage)

    transforms = transform_builder(args.image_height)

    train_dataset = VisdaTrainDataset(
        (storage_path/'train').resolve(), transform=transforms["contrast_train_transforms"], n_views=2)

    valid_dataset = VisdaValidDataset((storage_path/'validation').resolve(
    ), transform=transforms["contrast_valid_transforms"], n_views=2)

    unsupervised_dataset = VisdaUnsupervisedDataset(
        train_dataset, valid_dataset)

    create_samples(unsupervised_dataset, Path(args.transform_sample_directory))

    linear_train_data = ImageFolder(
        (storage_path/"train").resolve(), transform=transforms["linear_transform"])
    linear_validation_data = ImageFolder(
        (storage_path/"validation").resolve(), transform=transforms["linear_transform"])

    train_dataloader = DataLoader(
        unsupervised_dataset, args.pretrain_batch_size, num_workers=16,shuffle=True)

    if args.pretrained_weights_path:
        model = SimCLR.load_from_checkpoint(Path(args.pretrained_weights_path).resolve(),batch_size=args.pretrain_batch_size,warmup_epochs=0,num_samples=len(train_dataloader)*args.pretrain_batch_size,use_all_features=args.use_all_features)
    else:
        model = SimCLR(args.pretrain_batch_size, len(train_dataloader)*args.pretrain_batch_size,
            lr=args.pretrain_learning_rate,
            keep_mlp=args.keep_mlp,high_penalty_weight=args.high_penalty_weight,
            low_penalty_weight=args.low_penalty_weight,
            use_all_features=args.use_all_features,
            model_name=args.model_name)

    linear_seperablity_metric = SSLOnlineEvaluator(
        train_dataset=linear_train_data,
        valid_dataset=linear_validation_data,
        finetune_every_n_epoch=args.finetune_every_n_epoch,
        num_classes=len(linear_train_data.classes),
        batch_size=args.finetune_batchsize,
        epochs=args.finetune_epochs,
        encoder_dimension= model.encoder_dimension,
        finetune_percentage=args.pretrain_finetune_percentage
    )

    checkpoint = ModelCheckpoint(dirpath=args.checkpoint_directory, save_top_k=args.save_top_k_models, every_n_epochs=args.save_models_every_n_epoch,monitor="adaptation_acc_one_percent",
                                 filename='{epoch}-{adaptation_acc_one_percent:.2f}-{linear_train_acc:.2f}',mode="max")
    progress_bar = TQDMProgressBar()

    callbacks = [linear_seperablity_metric, checkpoint, progress_bar]

    # is the max_epoch argument necessary?

    tensor_logger_path = Path(args.log_directory)/'tensorboard'
    wandb_logger_path = Path(args.log_directory)/'wandb'
    trainer = pl.Trainer(callbacks=callbacks, accelerator="gpu", devices=args.num_gpus, strategy='dp',logger=[TensorBoardLogger(
        save_dir=tensor_logger_path), WandbLogger(save_dir=wandb_logger_path, project="SimCLR-VisDA")]
        , max_epochs=args.pretrain_epochs, log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_dataloader)
    # callbacks: save model (weights and biases?). linear seperablity metric. progress bar (weights and biases?).
if args.action == 'evaluate':
    storage_path = Path(args.storage)

    transforms = transform_builder(args.image_height)

    linear_train_data = ImageFolder(
        (storage_path/"train").resolve(), transform=transforms["linear_transform"])
    n_classes = len(linear_train_data.classes)

    len_same_dist_val = args.evaluation_same_dist_val_percentage * len(linear_train_data) // 100
    len_train = args.evaluation_train_percentage * len(linear_train_data) // 100
    same_dist_val_dataset, linear_train_data, _ = random_split(linear_train_data, [len_same_dist_val, len_train,len(linear_train_data) - len_same_dist_val - len_train])

    same_dist_val_dataloader = DataLoader(same_dist_val_dataset, args.finetune_batchsize, num_workers=16)

    linear_validation_data = ImageFolder(
        (storage_path/"validation").resolve(), transform=transforms["linear_transform"])

    train_dataloader = DataLoader(linear_train_data,args.finetune_batchsize,num_workers=16,shuffle=True)
    other_dist_valid_dataloader = DataLoader(linear_validation_data,args.finetune_batchsize,num_workers=16)

    simclr = SimCLR.load_from_checkpoint(Path(args.pretrained_weights_path).resolve(),batch_size=args.pretrain_batch_size,warmup_epochs=0,num_samples=len(train_dataloader),use_all_features=args.use_all_features)
    simclr.train(False)

    model = Evaluator(simclr,n_classes=n_classes,n_hidden=args.evaluator_hidden_dim,drop_p=args.drop_p)

    progress_bar = TQDMProgressBar()

    callbacks = [progress_bar]

    # is the max_epoch argument necessary?

    tensor_logger_path = Path(args.log_directory)/'tensorboard'
    wandb_logger_path = Path(args.log_directory)/'wandb'

    trainer = pl.Trainer(callbacks=callbacks, accelerator="gpu", devices=args.num_gpus, logger=[TensorBoardLogger(

        save_dir=tensor_logger_path), WandbLogger(save_dir=wandb_logger_path, project="SimCLR-VisDA")], max_epochs=args.finetune_epochs,
        # TODO: move this to argparse
        log_every_n_steps=1, strategy='dp')

    trainer.fit(model, train_dataloader, other_dist_valid_dataloader) #, same_dist_val_dataloader])