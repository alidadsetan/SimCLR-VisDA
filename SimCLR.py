import pytorch_lightning as pl
import torch
# from pl_bolts.models.self_supervised.resnets import resnet50_bn
from torchvision.models import resnet50, ResNet50_Weights
from pl_bolts.optimizers.lars import LARS

from torch.optim import Adam
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import nn
# from pl_bolts.models.self_supervised.evaluator import Flatten
# from pl_bolts.models.self_supervised import SimCLR as bolts_simclr
from torch.nn import functional as F
import timm


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 num_samples,
                #  weight_path,
                 keep_mlp,
                 high_penalty_weight,
                 low_penalty_weight,
                 mlp_dimension=2048,
                 warmup_epochs=0,
                 lr=1e-4,
                 opt_weight_decay=1e-6,
                 loss_temperature=0.5,
                 model_name='tv_resnet50',
                 **kwargs):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()
        # self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.encoder.fc = nn.Sequential()
        self.encoder = timm.create_model(model_name,pretrained=True,features_only=True)
        # self.encoder = bolts_simclr.load_from_checkpoint(weight_path,strict=False).encoder
        # self.encoder.eval()


        # h -> || -> z
        self.projection = Projection(output_dim=mlp_dimension)

    @property
    def encoder_dimension(self):
        # TODO: change this
        return 2048
        # if self.hparams.keep_mlp:
        #     return self.hparams.mlp_dimension
        # else:
        #     # return self.encoder.avgpool.output_size
        #     return 2048

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def configure_optimizers(self):
        # TRICK 1 (Use lars + filter weights)
        # exclude certain parameters
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        )

        optimizer = LARS(
            parameters,
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=1e-6,
            trust_coefficient=0.001)

        # Trick 2 (after each step)
        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        # if isinstance(x, list):
        #     x = x[0]

        result = self.encoder.forward_features(x)

        # added for testing
        # if self.hparams.keep_mlp:
        #     result = self.projection(result)
        # if isinstance(result, list):
        #     result = result[-1]
        return result

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # result = pl.TrainResult(minimize=loss)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # result = pl.EvalResult(checkpoint_on=loss)
        self.log('avg_val_loss', loss)
        return loss

    def shared_step(self, batch, batch_idx):
        (img1, img2), (labels,_) = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048)
        h1 = self.encoder.forward_features(img1)
        h2 = self.encoder.forward_features(img2)
        # if isinstance(h1,list):
        #     h1 = h1[-1]
        #     h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2 ,self.hparams.loss_temperature, labels=labels)

        return loss

    def compute_neg_weights(self,labels):
        column_labels = labels.expand((len(labels)), -1)
        row_labels = column_labels.t()
        
        no_label = (row_labels == -1) | (column_labels == -1)
        diagonal = torch.eye(len(labels),device=labels.device).bool()

        hight_penalty = (row_labels != column_labels) & ~no_label
        low_penalty = (row_labels == column_labels) & ~no_label 
        regular_panalty = (no_label & ~diagonal)

        quarter_result = self.hparams.high_penalty_weight * hight_penalty + self.hparams.low_penalty_weight * low_penalty + regular_panalty

        result_row_1 = torch.concat([quarter_result,quarter_result+torch.eye(len(labels),device=labels.device)],dim=1)
        result_row_2 = torch.concat([quarter_result+torch.eye(len(labels),device=labels.device), quarter_result],dim=1)

        return torch.concat([result_row_1,result_row_2],dim=0)

    def nt_xent_loss(self,out_1, out_2, temperature,labels):
        out = torch.cat([out_1, out_2], dim=0)

        # Full similarity matrix
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / temperature)

        # mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg_weights = self.compute_neg_weights(labels)

        neg = torch.mm(neg_weights,sim).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss
