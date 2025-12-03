import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import random
from network.losses.sam_aux_loss import ObjectConsistencyLoss, SAMBoundaryLoss  # 已导入


# 运行命令：python train.py -c config/loveda/d2ls.py

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


import pytorch_model_summary
from torchinfo import summary
import torchsummary


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss

        # === 新增：SAM 辅助损失函数 ===
        self.obj_loss_fn = ObjectConsistencyLoss(lambda_obj=getattr(config, 'lambda_obj', 0.5))
        self.bound_loss_fn = SAMBoundaryLoss(lambda_bound=getattr(config, 'lambda_bound', 0.5))

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        # === 新增：获取 SAM masks（可能为 None）===
        sgo_masks = batch.get('sgo')  # (B, H, W) LongTensor
        sgb = batch.get('sgb')  # (B, H, W) FloatTensor

        predictions = self.net(img)  # 可能返回 seg_output 或 (seg_output, boundary_pred)

        # === 处理模型多输出（主分割 + 可选边界头 + 可选 aux 头）===
        boundary_pred = None
        aux_pred = None

        if isinstance(predictions, (tuple, list)):
            if len(predictions) >= 2:
                # 我们新增的边界头在第2个位置（索引1）
                main_pred = predictions[0]
                boundary_pred = predictions[1]
                if len(predictions) > 2:
                    aux_pred = predictions[2]  # 如果还有其他头
            else:
                main_pred = predictions[0]
        else:
            main_pred = predictions

        # === 主损失 ===
        loss = self.loss(main_pred, mask)

        # === 原有 aux 损失（如果启用）===
        if getattr(self.config, 'use_aux_loss', False) and aux_pred is not None:
            aux_loss = self.loss(aux_pred, mask)
            loss = loss + 0.4 * aux_loss  # 典型权重 0.4

        # === 新增：SAM 辅助损失 ===
        aux_sam_loss = 0.0

        if sgo_masks is not None:
            obj_loss = self.obj_loss_fn(main_pred, sgo_masks)
            aux_sam_loss += obj_loss
            self.log('train/obj_loss', obj_loss, prog_bar=True, on_step=False, on_epoch=True)

        if boundary_pred is not None and sgb is not None:
            bound_loss = self.bound_loss_fn(boundary_pred, sgb)
            aux_sam_loss += bound_loss
            self.log('train/bound_loss', bound_loss, prog_bar=True, on_step=False, on_epoch=True)

        loss += aux_sam_loss
        self.log('train/sam_aux_loss', aux_sam_loss, prog_bar=True, on_step=False, on_epoch=True)

        # === 指标计算（使用主预测）===
        pre_mask = nn.Softmax(dim=1)(main_pred)
        pre_mask = pre_mask.argmax(dim=1)

        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        self.log('train/total_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'train_mIoU': mIoU,
                      'train_F1': F1,
                      'train_OA': OA}
        print('train:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        predictions = self.forward(img)

        # 验证时只取主预测
        if isinstance(predictions, (tuple, list)):
            pred = predictions[0]
        else:
            pred = predictions

        pre_mask = nn.Softmax(dim=1)(pred)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        loss_val = self.loss(pred, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        eval_value = {'val_mIoU': mIoU,
                      'val_F1': F1,
                      'val_OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('logs', name=config.log_name)
    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback],
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()