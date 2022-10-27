import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

class Evaluator(pl.LightningModule):
    def __init__(self, sim_clr,n_classes,n_hidden: int,drop_p=.1):
        super().__init__()
        self.sim_clr = sim_clr # .to(device)? .train(False)? inference? 
        self.evaluator = SSLEvaluator(n_input=self.sim_clr.encoder_dimension,n_classes=n_classes,p=drop_p,n_hidden=None if n_hidden == 0 else n_hidden)

    def shared_step(self, batch):
        with torch.no_grad():
            x, y = batch
            reps = self.sim_clr(x).flatten(start_dim=1)

        logits = self.evaluator(reps)
        loss = F.cross_entropy(logits,y)
        return loss
    
    def training_step(self, batch):
        loss = self.shared_step(batch)
        self.log('evaluation_batch_training_loss', loss)
        return loss
    
    def validation_step(self, batch):
        loss = self.shared_step(batch)
        self.log('evaluation_batch_validation_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer