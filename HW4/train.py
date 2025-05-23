import wandb
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from utils.loss_utils import edge_loss
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.val_utils import compute_psnr_ssim

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.edge_loss_fn = edge_loss
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

        # Forward pass
        restored = self.net(degrad_patch)

        l1 = self.loss_fn(restored, clean_patch)
        edge = self.edge_loss_fn(restored, clean_patch)
        loss = l1 + 0.1 * edge

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("l1_loss", l1, on_step=True, on_epoch=True, sync_dist=True)
        self.log("edge_loss", edge, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log PSNR and SSIM
        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)
        self.log("psnr", psnr, on_step=True, on_epoch=True, sync_dist=True)
        self.log("ssim", ssim, on_step=True, on_epoch=True, sync_dist=True)

        # Log images every N steps or only first few batches
        if self.logger and batch_idx % 6000 == 0:
            # Convert to grid
            grid = torchvision.utils.make_grid(torch.cat([
                degrad_patch[:4],  # input
                restored[:4],      # output
                clean_patch[:4],   # ground truth
            ], dim=0), nrow=4, normalize=True, scale_each=True)

            # Log to wandb
            self.logger.experiment.log({
                "Sample Input/Output/GT": wandb.Image(grid),
                "global_step": self.global_step,
                "epoch": self.current_epoch
            })
        
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger, name=opt.wandb_name)
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    ##### Dataset and DataLoader #####
    trainset = PromptTrainDataset(opt)
    # checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    checkpoint_callback = ModelCheckpoint(
        monitor="psnr",
        mode="max", 
        save_top_k=1,
        save_last=False, 
        dirpath=opt.ckpt_dir,
        filename=opt.ckpt_name 
    )

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    ##### Model #####
    model = PromptIRModel()
    
    ##### Training #####
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, \
                         strategy="ddp_find_unused_parameters_true", logger=logger, \
                         callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)

    ##### After training: print best PSNR #####
    print(f"✅ Best PSNR: {checkpoint_callback.best_model_score.item():.4f}")
    print(f"📦 Best model path: {checkpoint_callback.best_model_path}")
    
if __name__ == '__main__':
    main()
