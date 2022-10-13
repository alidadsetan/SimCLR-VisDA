from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        finetune_full_labels_every_n_epoch: int ,
        finetune_first_ten_percent_every_n_epoch: int,
        batch_size: int,
        epoches = 100,
        drop_p: float = 0.2,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = None
        self.dataset: Optional[str] = None
        self.num_classes: Optional[int] = num_classes
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self._recovered_callback_state: Optional[Dict[str, Any]] = None
        self.epochs = epoches

        self.completed_epoches = 0
        self.finetune_with_full_labels_every_n_epoch = finetune_full_labels_every_n_epoch 
        self.finetune_with_ten_percent_every_n_epoch = finetune_first_ten_percent_every_n_epoch

        first_ten_percent_length = len(train_dataset)//10

        self.ten_percent_train_loader = DataLoader(random_split(train_dataset, [first_ten_percent_length, len(train_dataset) - first_ten_percent_length])[0],
            # TODO: what the fuck is 8? at least some parameters
            batch_size=batch_size, shuffle=True,num_worker=8)

        self.full_train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8)
        self.validation_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes
        if self.dataset is None:
            self.dataset = trainer.datamodule.name

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.completed_epoches += 1
        if self.completed_epoches%self.finetune_with_full_labels_every_n_epoch == 0:
            t_loader = self.full_train_loader
        elif self.completed_epoches%self.finetune_with_ten_percent_every_n_epoch == 0:
            t_loader = self.ten_percent_train_loader
        else:
            return
        self.compute_adaptation_loss(trainer,pl_module,"epoch_end", t_loader)


    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.compute_adaptation_loss(trainer,pl_module, "fit_start", self.ten_percent_train_loader)

    def compute_adaptation_loss(self, trainer: Trainer, pl_module: LightningModule, source: str, t_loader: DataLoader) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            # output of previous
            n_input=2048,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=None,
        ).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if accel.use_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif accel.use_dp:
                from torch.nn.parallel import DataParallel as DP

                self.online_evbbkaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

        for i in range(self.epochs):
            if i == self.epochs - 1:
                train_accs = []
            for batch in t_loader:
                train_acc, mlp_loss = self.shared_step(pl_module, batch)
                if i == self.epochs - 1:
                    train_accs.append({"acc": train_acc, "num": (batch[1].shape)[0]})
                mlp_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if i == self.epochs - 1:
                epoch_train_acc = sum([x["acc"] * x["num"] for x in train_accs])/sum([x["num"] for x in train_accs])
                pl_module.log("linear_train_acc", epoch_train_acc)


        val_accs = []
        for batch in self.validation_loader:
            val_acc, mlp_loss = self.shared_step(pl_module, batch)   
            val_accs.append({"acc": val_acc, "num": (batch[1].shape)[0]})
        
        val_acc = sum([x["acc"] * x["num"] for x in val_accs])/sum([x["num"] for x in val_accs])
        pl_module.log("adaptation_acc_{}".format(source),val_acc)

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # inputs, y = batch
        x, y = batch

        # last input is for online eval
        # x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device)
                representations = pl_module(x).flatten(start_dim=1)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        acc = accuracy(mlp_logits.softmax(-1), y)

        return acc, mlp_loss

    # def on_train_batch_end(
    #     self,
    #     trainer: Trainer,
    #     pl_module: LightningModule,
    #     outputs: Sequence,
    #     batch: Sequence,
    #     batch_idx: int,
    # ) -> None:
    #     train_acc, mlp_loss = self.shared_step(pl_module, batch)

    #     # update finetune weights
    #     mlp_loss.backward()
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()

    #     pl_module.log("online_train_acc", train_acc, on_step=True, on_epoch=False)
    #     pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

    # def on_validation_batch_end(
    #     self,
    #     trainer: Trainer,
    #     pl_module: LightningModule,
    #     outputs: Sequence,
    #     batch: Sequence,
    #     batch_idx: int,
    #     dataloader_idx: int,
    # ) -> None:
    #     val_acc, mlp_loss = self.shared_step(pl_module, batch)
    #     pl_module.log("online_val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
    #     pl_module.log("online_val_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)