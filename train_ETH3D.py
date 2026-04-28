import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
from core.utils.utils import InputPadder
from core.monster import Monster
from core.loss import sequence_loss_stable
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import wandb
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image
from datetime import datetime


def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def get_padded_H_inv_simple(H_inv, pad_left, pad_top):
    """简化版本，假设H_inv是[1,3,3]的tensor"""
    device = H_inv.device
    dtype = H_inv.dtype

    # 创建变换矩阵 [1,3,3]
    T_pad_to_orig = torch.tensor([
        [[1, 0, -pad_left],
         [0, 1, -pad_top],
         [0, 0, 1]]
    ], device=device, dtype=dtype)

    T_orig_to_pad = torch.tensor([
        [[1, 0, pad_left],
         [0, 1, pad_top],
         [0, 0, 1]]
    ], device=device, dtype=dtype)

    # 使用torch.bmm进行批量矩阵乘法
    return torch.bmm(torch.bmm(T_orig_to_pad, H_inv), T_pad_to_orig)


def inverse_transform_verification1_fully_parallel(horizontal_disparity: torch.Tensor,
                                                   vertical_disparity: torch.Tensor,
                                                   H_inv: torch.Tensor) -> torch.Tensor:

    """
    完全并行化版本 - 在GPU上更快但使用更多内存
    """
    B, C, H, W = horizontal_disparity.shape
    device = horizontal_disparity.device
    dtype = torch.float32

    # 确保数据类型
    horizontal_disparity = horizontal_disparity.float()
    vertical_disparity = vertical_disparity.float()
    H_inv = H_inv.float()

    if H_inv.dim() == 2:
        H_inv = H_inv.unsqueeze(0).expand(B, -1, -1)

    # 创建坐标网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    # 扩展到batch维度
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # 去除channel维度
    h_disp = horizontal_disparity[:, 0]  # (B, H, W)
    v_disp = vertical_disparity[:, 0]  # (B, H, W)

    # 计算右图坐标
    right_x = grid_x - h_disp  # (B, H, W)
    right_y = grid_y - v_disp  # (B, H, W)

    # 重塑为批量矩阵乘法格式
    coords_flat = torch.stack([
        right_x.reshape(B, -1),
        right_y.reshape(B, -1),
        torch.ones(B, H * W, device=device, dtype=dtype)
    ], dim=1)  # (B, 3, H*W)

    # 批量矩阵乘法 - 一次处理所有batch
    coords_transformed = torch.bmm(H_inv, coords_flat)  # (B, 3, H*W)

    # 转换为非齐次坐标
    z_values = coords_transformed[:, 2, :].reshape(B, H, W)
    x_transformed = coords_transformed[:, 0, :].reshape(B, H, W) / (z_values + 1e-10)

    # 计算恢复的视差
    recovered_disp = grid_x - x_transformed

    # 应用有效性掩码
    valid_mask = (torch.abs(h_disp )> 0.0) | (torch.abs(v_disp) > 0.0)
    valid_mask &= (torch.abs(z_values) > 1e-15)
    valid_mask &= (recovered_disp > 0) & (recovered_disp < 400)

    # 创建输出
    recovered_disparity = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    recovered_disparity[:, 0] = torch.where(valid_mask, recovered_disp, torch.zeros_like(recovered_disp))

    return recovered_disparity





def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics

def sequence_loss_with_vertical(disp_preds, disp_init_pred, disp_gt, valid,
                                disp_preds_vertical, disp_init_pred_vertical, disp_gt_vertical, valid_vertical,
                                loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of horizontal and vertical disp predictions """
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    assert len(disp_preds_vertical) == n_predictions

    total_disp_loss = 0.0

    # =========================== 水平视差损失计算 ===========================
    # exclude invalid pixels and extremely large displacements
    mag_h = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid_h = ((valid >= 0.5) & (mag_h < max_disp)).unsqueeze(1)
    assert valid_h.shape == disp_gt.shape, [valid_h.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid_h.bool()]).any()

    # 初始预测损失 - 水平
    disp_loss_h = 1.0 * F.smooth_l1_loss(disp_init_pred[valid_h.bool()], disp_gt[valid_h.bool()], size_average=True)

    # 序列预测损失 - 水平
    for i in range(n_predictions):
        assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid_h.shape, [i_loss.shape, valid_h.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss_h += i_weight * i_loss[valid_h.bool()].mean()

    # =========================== 垂直视差损失计算 ===========================
    # exclude invalid pixels and extremely large displacements
    mag_v = torch.sum(disp_gt_vertical ** 2, dim=1).sqrt()
    valid_v = ((valid_vertical >= 0.5) & (mag_v < max_disp)).unsqueeze(1)
    assert valid_v.shape == disp_gt_vertical.shape, [valid_v.shape, disp_gt_vertical.shape]
    assert not torch.isinf(disp_gt_vertical[valid_v.bool()]).any()

    # 初始预测损失 - 垂直
    disp_loss_v = 1.0 * F.smooth_l1_loss(disp_init_pred_vertical[valid_v.bool()], disp_gt_vertical[valid_v.bool()],
                                         size_average=True)

    # 序列预测损失 - 垂直
    for i in range(n_predictions):
        assert not torch.isnan(disp_preds_vertical[i]).any() and not torch.isinf(disp_preds_vertical[i]).any()
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds_vertical[i] - disp_gt_vertical).abs()
        assert i_loss.shape == valid_v.shape, [i_loss.shape, valid_v.shape, disp_gt_vertical.shape,
                                               disp_preds_vertical[i].shape]
        disp_loss_v += i_weight * i_loss[valid_v.bool()].mean()

    # =========================== 合并总损失 ===========================
    total_disp_loss = disp_loss_h + disp_loss_v

    # =========================== 计算评估指标 ===========================
    # 水平视差指标
    epe_h = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe_h = epe_h.view(-1)[valid_h.view(-1)]

    metrics_h = {
        'epe_h': epe_h.mean(),
        '1px_h': (epe_h < 1).float().mean(),   #本来之前是.item()
        '3px_h': (epe_h < 3).float().mean(),
        '5px_h': (epe_h < 5).float().mean(),
    }

    # 垂直视差指标
    epe_v = torch.sum((disp_preds_vertical[-1] - disp_gt_vertical) ** 2, dim=1).sqrt()
    epe_v = epe_v.view(-1)[valid_v.view(-1)]

    metrics_v = {
        'epe_v': epe_v.mean(),
        '1px_v': (epe_v < 1).float().mean(),
        '3px_v': (epe_v < 3).float().mean(),
        '5px_v': (epe_v < 5).float().mean(),
    }

    # 合并所有指标
    metrics = {**metrics_h, **metrics_v}

    # 可选：添加总体指标
    epe_total = torch.cat([epe_h, epe_v])
    metrics.update({
        'epe_total': epe_total.mean(),
        '1px_total': (epe_total < 1).float().mean(),
        '3px_total': (epe_total < 3).float().mean(),
        '5px_total': (epe_total < 5).float().mean(),
    })

    return total_disp_loss, metrics


def sequence_loss_complete(disp_preds, disp_init_pred, disp_gt, valid,
                           disp_preds_vertical, disp_init_pred_vertical, disp_gt_vertical, valid_vertical,
                           disp_preds_rectified, disp_init_pred_rectified, disp_gt_rectified, valid_rectified,
                           loss_gamma=0.9, max_disp=192,use_adaptive_weights=True,loss_history=None):
    """
    Complete loss function with three components:
    1. Horizontal disparity loss (原始水平视差)
    2. Vertical disparity loss (原始垂直视差)
    3. Rectified disparity loss (极线矫正后的视差)
    """
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    assert len(disp_preds_vertical) == n_predictions
    assert len(disp_preds_rectified) == n_predictions

    total_disp_loss = 0.0

    # =========================== 水平视差损失计算 ===========================
    # exclude invalid pixels and extremely large displacements
    mag_h = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid_h = ((valid >= 0.5) & (mag_h < max_disp)).unsqueeze(1)
    assert valid_h.shape == disp_gt.shape, [valid_h.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid_h.bool()]).any()

    # 初始预测损失 - 水平
    #disp_loss_h = 1.0 * F.smooth_l1_loss(disp_init_pred[valid_h.bool()], disp_gt[valid_h.bool()], size_average=True)
    disp_loss_h =0.0
    # 序列预测损失 - 水平
    for i in range(n_predictions):
        assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid_h.shape, [i_loss.shape, valid_h.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss_h += i_weight * i_loss[valid_h.bool()].mean()

    # =========================== 垂直视差损失计算 ===========================
    # exclude invalid pixels and extremely large displacements
    mag_v = torch.sum(disp_gt_vertical ** 2, dim=1).sqrt()
    valid_v = ((valid_vertical >= 0.5) & (mag_v < max_disp)).unsqueeze(1)
    assert valid_v.shape == disp_gt_vertical.shape, [valid_v.shape, disp_gt_vertical.shape]
    assert not torch.isinf(disp_gt_vertical[valid_v.bool()]).any()

    # 初始预测损失 - 垂直
    has_valid_v = valid_v.bool().any()
    disp_loss_v = 0.0

    if has_valid_v:
        # 只有在有有效像素时才进行断言和损失计算
        assert not torch.isinf(disp_gt_vertical[valid_v.bool()]).any()

        # 初始预测损失 - 垂直
        init_valid_v = valid_v.bool() & ~torch.isnan(disp_init_pred_vertical)
        if init_valid_v.any():
            # disp_loss_v = 1.0 * F.smooth_l1_loss(
            #     disp_init_pred_vertical[init_valid_v],
            #     disp_gt_vertical[init_valid_v],
            #     reduction='mean'
            # )
            disp_loss_v = 0.0;

    #disp_loss_v = 1.0 * F.smooth_l1_loss(disp_init_pred_vertical[valid_v.bool()], disp_gt_vertical[valid_v.bool()],
    #                                     size_average=True)

    # 序列预测损失 - 垂直
        for i in range(n_predictions):
            assert not torch.isnan(disp_preds_vertical[i]).any() and not torch.isinf(disp_preds_vertical[i]).any()
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds_vertical[i] - disp_gt_vertical).abs()
            assert i_loss.shape == valid_v.shape, [i_loss.shape, valid_v.shape, disp_gt_vertical.shape,
                                                   disp_preds_vertical[i].shape]
            disp_loss_v += i_weight * i_loss[valid_v.bool()].mean()
    else:
        # 没有有效的垂直视差像素，使用小的常数损失避免NaN
        disp_loss_v = torch.tensor(0.0, device=disp_gt_vertical.device, dtype=disp_gt_vertical.dtype)

    # =========================== 极线矫正后视差损失计算 (新增部分) ===========================
    disp_loss_rectified = 0.0
    mag_rectified = torch.sum(disp_gt_rectified ** 2, dim=1).sqrt()
    valid_rect = ((valid_rectified >= 0.5) & (mag_rectified < max_disp)).unsqueeze(1)
    assert valid_rect.shape == disp_gt_rectified.shape, [valid_rect.shape, disp_gt_rectified.shape]
    assert not torch.isinf(disp_gt_rectified[valid_rect.bool()]).any()

    # 初始预测损失 - 极线矫正后
    init_valid_rect = valid_rect.bool() & ~torch.isnan(disp_init_pred_rectified)
    disp_loss_rectified += 1.0 * F.smooth_l1_loss(disp_init_pred_rectified[init_valid_rect],
                                                  disp_gt_rectified[init_valid_rect],
                                                  reduction='mean')

    # 序列预测损失 - 极线矫正后
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds_rectified[i] - disp_gt_rectified).abs()
        assert i_loss.shape == valid_rect.shape, [i_loss.shape, valid_rect.shape, disp_gt_rectified.shape,
                                                  disp_preds_rectified[i].shape]
        disp_loss_rectified += i_weight * i_loss[valid_rect.bool() & ~torch.isnan(i_loss)].mean()

    if use_adaptive_weights:
        # 初始化损失历史
        if loss_history is None:
            loss_history = {}

        # 更新损失历史（使用指数移动平均）
        momentum = 0.9
        with torch.no_grad():
            current_losses = {
                'h': disp_loss_h.item(),
                'v': disp_loss_v.item(),
                'r': disp_loss_rectified.item()
            }

            if 'avg_h' not in loss_history:
                # 首次初始化
                loss_history['avg_h'] = current_losses['h']
                loss_history['avg_v'] = current_losses['v']
                loss_history['avg_r'] = current_losses['r']
            else:
                # 更新移动平均
                loss_history['avg_h'] = momentum * loss_history['avg_h'] + (1 - momentum) * current_losses['h']
                loss_history['avg_v'] = momentum * loss_history['avg_v'] + (1 - momentum) * current_losses['v']
                loss_history['avg_r'] = momentum * loss_history['avg_r'] + (1 - momentum) * current_losses['r']

            # 计算自适应权重，使加权后的损失在同一量级
            base_scale = loss_history['avg_h']  # 以水平视差为基准
            w_h = 1.0
            w_v = base_scale / (loss_history['avg_v'] + 1e-8)
            w_r = base_scale / (loss_history['avg_r'] + 1e-8)

            # 限制权重范围，避免极端值
            w_v = torch.clamp(torch.tensor(w_v), 0.05, 2.0).item()
            w_r = torch.clamp(torch.tensor(w_r), 0.1, 2.0).item()

    # =========================== 合并总损失 ===========================
    total_disp_loss = w_h*disp_loss_h + w_v*disp_loss_v + w_r*disp_loss_rectified

    # =========================== 计算评估指标 ===========================
    # 水平视差指标
    epe_h = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe_h = epe_h.view(-1)[valid_h.view(-1)]

    metrics_h = {
        'epe_h': epe_h.mean(),
        '1px_h': (epe_h < 1).float().mean(),
        '3px_h': (epe_h < 3).float().mean(),
        '5px_h': (epe_h < 5).float().mean(),
    }

    # 垂直视差指标
    epe_v = torch.sum((disp_preds_vertical[-1] - disp_gt_vertical) ** 2, dim=1).sqrt()
    epe_v = epe_v.view(-1)[valid_v.view(-1)]

    metrics_v = {
        'epe_v': epe_v.mean(),
        '1px_v': (epe_v < 1).float().mean(),
        '3px_v': (epe_v < 3).float().mean(),
        '5px_v': (epe_v < 5).float().mean(),
    }

    # 极线矫正后视差指标 (新增部分)
    epe_rectified = torch.sum((disp_preds_rectified[-1] - disp_gt_rectified) ** 2, dim=1).sqrt()
    epe_rectified = epe_rectified.view(-1)[valid_rect.view(-1)]

    if valid_rect.bool().sum() == 0:
        epe_rectified = torch.Tensor([0.0]).cuda()

    metrics_rectified = {
        'epe_rectified': epe_rectified.mean(),
        '1px_rectified': (epe_rectified < 1).float().mean(),
        '3px_rectified': (epe_rectified < 3).float().mean(),
        '5px_rectified': (epe_rectified < 5).float().mean(),
    }

    # 合并所有指标
    metrics = {**metrics_h, **metrics_v, **metrics_rectified}

    # 添加总体指标
    epe_total = torch.cat([epe_h, epe_v])
    metrics.update({
        'epe_total': epe_total.mean(),
        '1px_total': (epe_total < 1).float().mean(),
        '3px_total': (epe_total < 3).float().mean(),
        '5px_total': (epe_total < 5).float().mean(),
    })

    # 添加包含极线矫正的完整总体指标
    if valid_rect.bool().sum() > 0:
        epe_all = torch.cat([epe_h, epe_v, epe_rectified])
        metrics.update({
            'epe_all': epe_all.mean(),
            '1px_all': (epe_all < 1).float().mean(),
            '3px_all': (epe_all < 3).float().mean(),
            '5px_all': (epe_all < 5).float().mean(),
            'total_loss': total_disp_loss,
            'disp_h_loss': disp_loss_h,
            'disp_v_loss': disp_loss_v,
            'disp_rectified_loss': disp_loss_rectified,
        })

    return total_disp_loss, metrics



def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr / 2.0, args.lr], args.total_step + 100,
    #                                           pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_eth3d')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(mixed_precision='bf16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    # accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    accelerator = Accelerator(mixed_precision='bf16',
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
                              project_dir='./logs2',
                              log_with='tensorboard', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name)


    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)

    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)
    model = Monster(cfg)
    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
    del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    while should_keep_training:
        active_train_loader = train_loader
        model.train()
        #model.module.freeze_bn()
        if hasattr(model, 'module'):
            model.module.freeze_bn()  # 多GPU情况
        else:
            model.freeze_bn()  # 单GPU情况   # 单个GPU修改这个位置。
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt_h, valid_h,disp_gt_v,valid_v,disp_gt,valid,H_invs,H_inv_ls= [x for x in data]
            with accelerator.autocast():
                disp_init_pred, disp_init_pred_vertical,disp_init_recover,disp_preds, disp_preds_vertical,disp_preds_gt,depth_mono = model(left, right, H_invs,iters=cfg.train_iters)
            loss, metrics = sequence_loss_stable(disp_preds, disp_init_pred, disp_gt_h, valid_h,
                                                   disp_preds_vertical,disp_init_pred_vertical, disp_gt_v, valid_v,
                                                   disp_preds_gt,disp_init_recover,disp_gt,valid)#,max_disp=cfg.max_disp)
            # open("logs/val_loss3.txt", "a").write(
            #     f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  total_loss {loss:.4f}\n")
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)

            ####visualize the depth_mono and disp_preds
            if total_step % 200 == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))


                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np_h = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np_h = gray_2_colormap_np(disp_gt_h[0].squeeze())
                disp_preds_np_v = gray_2_colormap_np(disp_preds_vertical[-1][0].squeeze())
                disp_gt_np_v = gray_2_colormap_np(disp_gt_v[0].squeeze())
                disp_preds_np_gt = gray_2_colormap_np(disp_preds_gt[-1][0].squeeze())
                disp_gt_np_gt = gray_2_colormap_np(disp_gt[0].squeeze())

                # os.makedirs("./vis", exist_ok=True)
                # Image.fromarray(depth_mono_np).save(f"vis/depth_{total_step}.png")
                # Image.fromarray(disp_preds_np_h).save(f"vis/pred_h_{total_step}.png")
                # Image.fromarray(disp_gt_np_h).save(f"vis/gt_h_{total_step}.png")
                # Image.fromarray(disp_preds_np_v).save(f"vis/pred_v_{total_step}.png")
                # Image.fromarray(disp_gt_np_v).save(f"vis/gt_v_{total_step}.png")
                # Image.fromarray(disp_preds_np_gt).save(f"vis/pred_gt_{total_step}.png")
                # Image.fromarray(disp_gt_np_gt).save(f"vis/gt_{total_step}.png")



                # disp_preds_tensor_h = torch.from_numpy(disp_preds_np_h).permute(2, 0, 1).float() / 255.0
                # disp_gt_tensor_h = torch.from_numpy(disp_gt_np_h).permute(2, 0, 1).float() / 255.0
                # disp_preds_tensor_v = torch.from_numpy(disp_preds_np_v).permute(2, 0, 1).float() / 255.0
                # disp_gt_tensor_v = torch.from_numpy(disp_gt_np_v).permute(2, 0, 1).float() / 255.0
                # disp_preds_tensor_gt = torch.from_numpy(disp_preds_np_gt).permute(2, 0, 1).float() / 255.0
                # disp_gt_tensor_gt = torch.from_numpy(disp_gt_np_gt).permute(2, 0, 1).float() / 255.0
                # depth_mono_tensor = torch.from_numpy(depth_mono_np).permute(2, 0, 1).float() / 255.0



                # accelerator.log({"disp_pred": disp_preds_tensor}, total_step)
                # accelerator.log({"disp_gt": disp_gt_tensor}, total_step)
                # accelerator.log({"depth_mono": depth_mono_tensor}, total_step)

                # accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                # accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                # accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)

            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save
        
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                torch.cuda.empty_cache()
                model.eval()
                # 存放分布式聚合前的本进程统计
                epe_h_all, out_h_all = [], []
                epe_v_all, out_v_all = [], []
                epe_r_all, out_r_all = [], []

                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    (imageL_file, imageR_file, _, _, GT_file), left, right, \
                        disp_gt_h, valid_h, disp_gt_v, valid_v, disp_gt, valid, H_invs, H_inv_ls = [x for x in data]

                    # 记录原尺寸
                    B, _, h_orig, w_orig = left.shape
                    padder = InputPadder(left.shape, divis_by=32)
                    left_p, right_p = padder.pad(left, right)

                    # 计算真实 pad 左上
                    h_padded, w_padded = left_p.shape[-2:]
                    pad_left = (w_padded - w_orig) // 2
                    pad_top = (h_padded - h_orig) // 2

                    # 仅用于网络前向的H_inv（考虑pad）
                    H_inv_padded = get_padded_H_inv_simple(H_inv_ls, pad_left, pad_top)

                    with torch.no_grad():
                        disp_pred_p, disp_pred_v_p, _ = model(left_p, right_p, H_inv_padded, iters=cfg.valid_iters,
                                                              test_mode=True)

                    # 去pad
                    disp_pred = padder.unpad(disp_pred_p)
                    disp_pred_vertical = padder.unpad(disp_pred_v_p)

                    # 用“原始的H_inv_ls+未pad的视差”做逆变换（得到rectified）
                    pred_ecover = inverse_transform_verification1_fully_parallel(disp_pred, disp_pred_vertical,
                                                                                 H_inv_ls)

                    # 断言尺寸
                    assert disp_pred.shape == disp_gt_h.shape, (disp_pred.shape, disp_gt_h.shape)
                    assert disp_pred_vertical.shape == disp_gt_v.shape, (disp_pred_vertical.shape, disp_gt_v.shape)
                    assert pred_ecover.shape == disp_gt.shape, (pred_ecover.shape, disp_gt.shape)

                    # 读取/准备遮挡掩码（建议放到Dataset）
                    occ_mask_np = np.array(Image.open(GT_file[0].replace('disp0GT.pfm', 'mask0nocc.png')))
                    occ_mask = torch.from_numpy(occ_mask_np).to(accelerator.device)

                    # 计算 EPE（通道维C=1）
                    epe_h = (disp_pred - disp_gt_h).abs().squeeze(1)  # [B,H,W]
                    epe_v = (disp_pred_vertical - disp_gt_v).abs().squeeze(1)
                    epe_r = (pred_ecover - disp_gt).abs().squeeze(1)

                    # 有效像素mask
                    val_mask_h = ((valid_h >= 0.5) & (occ_mask == 255)).flatten()
                    val_mask_v = ((valid_v >= 0.5) & (occ_mask == 255)).flatten()
                    val_mask_r = ((valid >= 0.5) & (occ_mask == 255)).flatten()

                    # 逐图统计（B通常=1，也兼容>1）
                    for b in range(B):
                        epe_h_img = epe_h[b].flatten()
                        epe_v_img = epe_v[b].flatten()
                        epe_r_img = epe_r[b].flatten()

                        vh = val_mask_h[b * h_orig * w_orig:(b + 1) * h_orig * w_orig]
                        vv = val_mask_v[b * h_orig * w_orig:(b + 1) * h_orig * w_orig]
                        vr = val_mask_r[b * h_orig * w_orig:(b + 1) * h_orig * w_orig]

                        if vh.any():
                            epe_h_all.append(epe_h_img[vh].mean())
                            out_h_all.append((epe_h_img[vh] > 1.0).float().mean())
                        if vv.any():
                            epe_v_all.append(epe_v_img[vv].mean())
                            out_v_all.append((epe_v_img[vv] > 1.0).float().mean())
                        if vr.any():
                            epe_r_all.append(epe_r_img[vr].mean())
                            out_r_all.append((epe_r_img[vr] > 1.0).float().mean())

                # 转成tensor并做分布式聚合
                def _gather_mean(x_list):
                    if len(x_list) == 0:
                        x = torch.tensor([0.0], device=accelerator.device)
                    else:
                        x = torch.stack(x_list) if isinstance(x_list[0], torch.Tensor) else torch.stack(
                            [torch.tensor(v, device=accelerator.device) for v in x_list])
                    x_all = accelerator.gather_for_metrics(x)  # [N_all]
                    return x_all.mean().item() if x_all.numel() > 0 else 0.0

                epe_h_mean = _gather_mean(epe_h_all)
                d1_h_mean = 100.0 * _gather_mean(out_h_all)
                epe_v_mean = _gather_mean(epe_v_all)
                d1_v_mean = 100.0 * _gather_mean(out_v_all)
                epe_r_mean = _gather_mean(epe_r_all)
                d1_r_mean = 100.0 * _gather_mean(out_r_all)

                if accelerator.is_main_process:
                    os.makedirs("logs_txt", exist_ok=True)
                    open("logs/ETH3D6.txt", "a").write(
                        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Step {total_step}: "
                        f"EPE {epe_r_mean:.4f}, D1 {d1_r_mean:.4f}, "
                        f"EPE_h {epe_h_mean:.4f}, D1_h {d1_h_mean:.4f}, "
                        f"EPE_v {epe_v_mean:.4f}, D1_v {d1_v_mean:.4f}\n"
                    )
                    print(f"Validation: EPE {epe_r_mean:.4f}, D1 {d1_r_mean:.4f}, "
                          f"EPE_h {epe_h_mean:.4f}, D1_h {d1_h_mean:.4f}, "
                          f"EPE_v {epe_v_mean:.4f}, D1_v {d1_v_mean:.4f}")

                model.train()
                #model.module.freeze_bn()
                if hasattr(model, 'module'):
                    model.module.freeze_bn()  # 多GPU情况
                else:
                    model.freeze_bn()  # 单GPU情况   # 单个GPU修改这个位置。

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save
    
    accelerator.end_training()

if __name__ == '__main__':
    main()