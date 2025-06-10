import os
from os import path as osp
from argparse import Namespace

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


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

        self.criterion = nn.L1Loss()

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
        signal_lq, signal_hq, _ = train_batch
        output = self(signal_lq)
        loss = self.criterion(output, signal_hq)

        self.log(
            "train/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        signal_lq, signal_hq = val_batch
        output = self(signal_lq)
        loss = self.criterion(output, signal_hq).detach()

        self.log(
            "val/loss",
            loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # # validation images for tensorboard visualization
        # val_img_name = "val/val_image_{}".format(filename[0].split("/")[-1])
        # grid = make_grid(output / 255)
        # self.logger.experiment.add_image(val_img_name, grid, self.trainer.current_epoch)

    def on_test_start(self):
        from src.utils.utils_logger import get_logger
        from src.utils.utils_saver import Saver

        assert self.task in ["sr", "car"]

        self.data_test = self.args.data_test

        self.save_dir_images = osp.join(
            self.save_path, "images", "results-{}".format(self.data_test)
        )
        if not osp.exists(self.save_dir_images):
            os.makedirs(self.save_dir_images, exist_ok=True)

        self.saver = Saver()
        self.saver.begin_background()

        self.text_logger = get_logger(log_path=osp.join(self.save_path, "result.log"))

        self.text_logger.info(f"Test dataset: {self.data_test}")
        self.text_logger.info(f"Scale factor: {self.scale}")

        self.border = self.scale if self.task == "sr" else 0

        self.avg_psnr = []

    # def test_step(self, batch, batch_idx):
    #     img_lq, img_gt, filename = batch

    #     if self.self_ensemble:
    #         # x8 self-ensemble
    #         output = self.forward_x8(img_lq, self.forward_chop)
    #     else:
    #         output = self.forward_chop(img_lq)

    #     output = quantize(output, self.rgb_range)

    #     psnr = calc_psnr(output, img_gt, self.scale, self.rgb_range)

    #     self.text_logger.info(f"Filename: {filename[0]} | PSNR: {psnr:.3f}")
    #     self.avg_psnr.append(psnr)

    #     self.saver.save_results(
    #         save_dir=self.save_dir_images,
    #         filename=filename[0],
    #         save_list=[output, img_lq, img_gt],
    #         scale=self.scale,
    #         rgb_range=self.rgb_range,
    #     )
