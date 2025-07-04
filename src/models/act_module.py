import os
from os import path as osp
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from src.utils import plot_signals


class ACTLitModule(LightningModule):
    def __init__(self, net: nn.Module, args: Namespace):
        super().__init__()

        self.args = args

        self.net = net

        self.task = args.task

        self.scale = args.scale
        self.crop_batch_size = args.crop_batch_size
        self.patch_size = args.patch_size

        # optimization configs
        self.lr = args.lr
        self.step_size = args.decay
        self.gamma = args.gamma
        self.betas = args.betas
        self.eps = args.epsilon
        self.weight_decay = args.weight_decay

        self.save_path = args.save_path

        # self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        signal_lq, signal_hq = train_batch
        output = self(signal_lq)
        loss = self.criterion(output, signal_hq)

        self.log(
            "train/loss",
            loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        signal_lq, signal_hq = val_batch
        output = self(signal_lq)
        loss = self.criterion(output, signal_hq).detach()

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # # validation images for tensorboard visualization
        # val_img_name = "val/val_image_{}".format(filename[0].split("/")[-1])
        # grid = make_grid(output / 255)
        # self.logger.experiment.add_image(val_img_name, grid, self.trainer.current_epoch)

    # def on_test_start(self):
    #     from src.utils.utils_logger import get_logger
    #     from src.utils.utils_saver import Saver

    #     assert self.task in ["sr", "car"]

    #     self.data_test = self.args.data_test

    #     self.save_dir_images = osp.join(
    #         self.save_path, "images", "results-{}".format(self.data_test)
    #     )
    #     if not osp.exists(self.save_dir_images):
    #         os.makedirs(self.save_dir_images, exist_ok=True)

    #     self.saver = Saver()
    #     self.saver.begin_background()

    #     self.text_logger = get_logger(log_path=osp.join(self.save_path, "result.log"))

    #     self.text_logger.info(f"Test dataset: {self.data_test}")
    #     self.text_logger.info(f"Scale factor: {self.scale}")

    #     self.border = self.scale if self.task == "sr" else 0

    #     self.avg_psnr = []

    def test_step(self, batch, batch_idx):
        signal_lq, signal_hq = batch

        output = self(signal_lq).squeeze()

        fig = plot_signals(
            low_res_signals=signal_lq.squeeze(),
            high_res_signals=signal_hq.squeeze(),
            model_signals=output,
            show=False,
            rows=1
        )
        filepath = f"{self.save_path}/{batch_idx}.png"
        fig.savefig(filepath, format="png")
        
        print(f"Saved figure {batch_idx}", end="\r")
