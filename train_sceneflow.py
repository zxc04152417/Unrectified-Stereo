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

import time
import torch.nn as nn
from contextlib import nullcontext

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    profile = None
    clever_format = None



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

def inverse_transform_verification1_fully_parallel(
    h: torch.Tensor,          # [B,1,H,W]
    v: torch.Tensor,          # [B,1,H,W]
    H_inv: torch.Tensor,      # [B,3,3] 或 [3,3]；要求像素中心系
    dmin: float = 0.0,
    dmax: float = 400.0,
    eps: float  = 1e-12,
) -> torch.Tensor:
    """
    返回：rectified 水平视差 d_rect，形状 [B,1,H,W]
    """
    assert h.dim() == 4 and v.dim() == 4, "h, v 必须是 4D (B,1,H,W)"
    B, Ch, H, W = h.shape
    assert v.shape == (B, Ch, H, W), "h 与 v 的形状不一致"
    device = h.device
    dtype  = torch.float32

    # 统一到 float32
    h = h.to(dtype)
    v = v.to(dtype)

    # 只取第1通道
    hx = h[:, 0]  # [B,H,W]
    vy = v[:, 0]  # [B,H,W]

    # H_inv 扩展到 batch
    if H_inv.dim() == 2:
        H_inv = H_inv.unsqueeze(0).expand(B, -1, -1)
    assert H_inv.shape == (B, 3, 3), "H_inv 必须是 [B,3,3] 或 [3,3]"

    # 网格（角点系）
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    # 原始右图坐标（角点系）→ 加 0.5 进像素中心系
    rx_c = (xx - hx) + 0.5   # [B,H,W]
    ry_c = (yy - vy) + 0.5

    # batched 齐次变换
    ones = torch.ones_like(rx_c)
    P = torch.stack([rx_c.reshape(B, -1), ry_c.reshape(B, -1), ones.reshape(B, -1)], dim=1)  # [B,3,HW]
    Q = torch.bmm(H_inv.to(dtype), P)  # [B,3,HW]

    # 非齐次（中心系）
    Q0 = Q[:, 0, :].reshape(B, H, W)
    Q1 = Q[:, 1, :].reshape(B, H, W)  # 目前未用到，保留以便调试
    Q2 = Q[:, 2, :].reshape(B, H, W)

    # 避免除零，保留符号
    denom = Q2.sign() * torch.clamp(Q2.abs(), min=eps)
    x_c = Q0 / denom                   # 仍在中心系
    x_rect = x_c - 0.5                 # 回到角点系

    # rectified 下的水平视差
    d_rect = xx - x_rect               # [B,H,W]

    # 有效像素：(|h|+|v|)>0 且 z 有效 且 视差范围合理
    hv_mag = h.abs().sum(dim=1) + v.abs().sum(dim=1)     # [B,H,W]
    valid = (hv_mag > 0) & (Q2.abs() > eps) & (d_rect > dmin) & (d_rect < dmax)

    # 写回输出
    out = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    out[:, 0] = torch.where(valid, d_rect, torch.zeros_like(d_rect))
    return out

#上面这个转化回来的可能需要改变，现在有新的改变函数




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

@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')   #这个代表的是config文件
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
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass',things_test=True)
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

                # —— 按图均权：本进程累加“图均值之和”和“有效图数” —— #
                epe_h_imgmean_sum_local = torch.tensor(0.0, device=accelerator.device)
                epe_v_imgmean_sum_local = torch.tensor(0.0, device=accelerator.device)
                epe_r_imgmean_sum_local = torch.tensor(0.0, device=accelerator.device)

                d1_h_imgmean_sum_local = torch.tensor(0.0, device=accelerator.device)
                d1_v_imgmean_sum_local = torch.tensor(0.0, device=accelerator.device)
                d1_r_imgmean_sum_local = torch.tensor(0.0, device=accelerator.device)

                img_count_h_local = torch.tensor(0, device=accelerator.device, dtype=torch.long)
                img_count_v_local = torch.tensor(0, device=accelerator.device, dtype=torch.long)
                img_count_r_local = torch.tensor(0, device=accelerator.device, dtype=torch.long)

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
                        disp_pred_p, disp_pred_v_p, _ = model(
                            left_p, right_p, H_inv_padded, iters=cfg.valid_iters, test_mode=True
                        )

                    # 去pad
                    disp_pred = padder.unpad(disp_pred_p)  # [B,1,H,W]
                    disp_pred_vertical = padder.unpad(disp_pred_v_p)  # [B,1,H,W]

                    # 用“原始的H_inv_ls+未pad的视差”做逆变换（得到rectified）
                    pred_ecover = inverse_transform_verification1_fully_parallel(
                        disp_pred, disp_pred_vertical, H_inv_ls
                    )  # 期望 [B,1,H,W]

                    # 断言尺寸
                    assert disp_pred.shape == disp_gt_h.shape, (disp_pred.shape, disp_gt_h.shape)
                    assert disp_pred_vertical.shape == disp_gt_v.shape, (disp_pred_vertical.shape, disp_gt_v.shape)
                    assert pred_ecover.shape == disp_gt.shape, (pred_ecover.shape, disp_gt.shape)

                    # 逐像素绝对误差（视差C=1）
                    epe_h = (disp_pred - disp_gt_h).abs().squeeze(1)  # [B,H,W]
                    epe_v = (disp_pred_vertical - disp_gt_v).abs().squeeze(1)  # [B,H,W]
                    epe_r = (pred_ecover - disp_gt).abs().squeeze(1)  # [B,H,W]

                    # 有效像素：valid>=0.5 且 |gt|<192（与你“上面那种”一致）
                    mask_h = ((valid_h >= 0.5) & (disp_gt_h.abs() < 192)).squeeze(1)  # [B,H,W], bool
                    mask_v = ((valid_v >= 0.5) & (disp_gt_v.abs() < 192)).squeeze(1)
                    mask_r = ((valid >= 0.5) & (disp_gt.abs() < 192)).squeeze(1)

                    # bad-3（如果你要 bad-1 就把阈值改为 1.0）
                    bad_h = (epe_h > 3.0)
                    bad_v = (epe_v > 3.0)
                    bad_r = (epe_r > 3.0)

                    # —— 按图统计：对每张图先做掩码均值，再在本进程累加 —— #
                    def per_image_means(epe_map, bad_map, mask):
                        B = epe_map.shape[0]
                        pix = mask.view(B, -1).sum(dim=1)  # [B]
                        # 为避免浮点溢出/NaN，先做和再除像素数
                        epe_sum = (epe_map * mask).view(B, -1).sum(dim=1)  # [B]
                        bad_sum = (bad_map.float() * mask).view(B, -1).sum(dim=1)  # [B]
                        valid_img = (pix > 0)
                        epe_mean = torch.zeros_like(epe_sum, dtype=torch.float)
                        d1_mean = torch.zeros_like(bad_sum, dtype=torch.float)
                        epe_mean[valid_img] = epe_sum[valid_img] / pix[valid_img].float().clamp_min(1)
                        d1_mean[valid_img] = bad_sum[valid_img].float() / pix[valid_img].float().clamp_min(1)
                        return epe_mean, d1_mean, valid_img

                    epe_h_imgmean, d1_h_imgmean, valid_img_h = per_image_means(epe_h, bad_h, mask_h)
                    epe_v_imgmean, d1_v_imgmean, valid_img_v = per_image_means(epe_v, bad_v, mask_v)
                    epe_r_imgmean, d1_r_imgmean, valid_img_r = per_image_means(epe_r, bad_r, mask_r)

                    # 本进程累计“图均值之和”和“有效图数量”
                    epe_h_imgmean_sum_local += epe_h_imgmean[valid_img_h].sum()
                    epe_v_imgmean_sum_local += epe_v_imgmean[valid_img_v].sum()
                    epe_r_imgmean_sum_local += epe_r_imgmean[valid_img_r].sum()

                    d1_h_imgmean_sum_local += d1_h_imgmean[valid_img_h].sum()
                    d1_v_imgmean_sum_local += d1_v_imgmean[valid_img_v].sum()
                    d1_r_imgmean_sum_local += d1_r_imgmean[valid_img_r].sum()

                    img_count_h_local += valid_img_h.long().sum()
                    img_count_v_local += valid_img_v.long().sum()
                    img_count_r_local += valid_img_r.long().sum()

                # —— 分布式聚合：把各进程“图均值之和 / 图数”收集并相加 —— #
                epe_h_all, epe_v_all, epe_r_all, \
                    d1_h_all, d1_v_all, d1_r_all, \
                    cnt_h_all, cnt_v_all, cnt_r_all = accelerator.gather_for_metrics((
                    epe_h_imgmean_sum_local, epe_v_imgmean_sum_local, epe_r_imgmean_sum_local,
                    d1_h_imgmean_sum_local, d1_v_imgmean_sum_local, d1_r_imgmean_sum_local,
                    img_count_h_local, img_count_v_local, img_count_r_local
                ))

                # 只在主进程计算全局均值与打印/写日志
                if accelerator.is_main_process:
                    epe_h_sum = epe_h_all.sum().item();
                    d1_h_sum = d1_h_all.sum().item();
                    n_h = cnt_h_all.sum().item()
                    epe_v_sum = epe_v_all.sum().item();
                    d1_v_sum = d1_v_all.sum().item();
                    n_v = cnt_v_all.sum().item()
                    epe_r_sum = epe_r_all.sum().item();
                    d1_r_sum = d1_r_all.sum().item();
                    n_r = cnt_r_all.sum().item()

                    eps = 1e-9
                    epe_h_mean = epe_h_sum / (n_h + eps);
                    d1_h_mean = 100.0 * (d1_h_sum / (n_h + eps))
                    epe_v_mean = epe_v_sum / (n_v + eps);
                    d1_v_mean = 100.0 * (d1_v_sum / (n_v + eps))
                    epe_r_mean = epe_r_sum / (n_r + eps);
                    d1_r_mean = 100.0 * (d1_r_sum / (n_r + eps))

                    os.makedirs("logs_txt", exist_ok=True)
                    open("logs/sceneflow_daxiu1.txt", "a").write(
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