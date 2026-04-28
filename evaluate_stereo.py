from __future__ import print_function, division
import sys
sys.path.append('core')
import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from monster import Monster, autocast
from pathlib import Path
import stereo_datasets as datasets
from utils.utils import InputPadder
from PIL import Image
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.ndim == 4 and t.shape[0] == 1:  # [1,C,H,W] -> [C,H,W]
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[0] == 1:  # [1,H,W]   -> [H,W]
            t = t.squeeze(0)
        return t.numpy()
    elif isinstance(t, np.ndarray):
        return t
    else:
        raise TypeError(f"Unsupported type: {type(t)}")

def _as_hwc(arr):
    # 支持 [C,H,W] / [H,W] / [H,W,3]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # [C,H,W]
        arr = np.transpose(arr, (1, 2, 0))        # -> [H,W,C]
    if arr.ndim == 2:                              # 灰度 -> 伪 3 通道
        arr = np.stack([arr]*3, axis=-1)
    return arr

def _normalize_rgb(arr):
    arr = arr.astype(np.float32)
    if arr.max() > 1.0 or arr.min() < 0.0:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lo = np.percentile(finite, 1)
            hi = np.percentile(finite, 99)
            if hi <= lo:
                lo = float(finite.min())
                hi = float(max(finite.max(), lo + 1e-6))
            arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr

def _save_rgb(path, img):
    arr = _to_numpy(img)
    arr = _as_hwc(arr)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = _normalize_rgb(arr)
    plt.imsave(path, arr)

def _save_disp(path, disp, cmap='turbo', robust=False, valid_mask=None):
    """
    保存视差；robust=True 时，用有效像素的 1%~99% 分位数做 vmin/vmax，显结构更清楚。
    valid_mask: (H,W) bool，限定参与统计的像素（如 valid_v 或 isfinite(GT)）。
    """
    d = _to_numpy(disp).astype(np.float32)
    if d.ndim == 3 and d.shape[0] == 1:
        d = d.squeeze(0)

    # 构造有效像素掩码
    if valid_mask is None:
        M = np.isfinite(d)
    else:
        vm = _to_numpy(valid_mask).astype(bool)
        M = np.isfinite(d) & vm

    d_vis = d.copy()
    d_vis = np.nan_to_num(d_vis, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if robust:
        finite_vals = d[M]
        if finite_vals.size > 0:
            vmin = np.percentile(finite_vals, 1)
            vmax = np.percentile(finite_vals, 99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(finite_vals))
                vmax = float(max(np.nanmax(finite_vals), vmin + 1e-6))
        else:
            vmin, vmax = float(np.nanmin(d_vis)), float(np.nanmax(d_vis))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0
        # 将无效像素压到 vmin，避免视觉污染
        d_plot = d_vis.copy()
        d_plot[~M] = vmin
        plt.imsave(path, d_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        # 直接 min→max（简单但可能“发一块色”）
        plt.imsave(path, d_vis, cmap=cmap)

def _save_disp_grad(path, disp):
    """保存视差梯度幅值图（灰度），结构通常更明显。"""
    d = _to_numpy(disp).astype(np.float32)
    if d.ndim == 3 and d.shape[0] == 1:
        d = d.squeeze(0)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(d)
    mag = np.hypot(gx, gy)
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    img = np.clip((mag - lo) / (hi - lo + 1e-6), 0, 1)
    plt.imsave(path, img, cmap='gray')

# ---------- main API ----------
def save_stereo_results(save_root, idx,
                        left, right,
                        disp_pred_h, disp_pred_v,disp_pred,
                        disp_gt_h, disp_gt_v, disp_gt,
                        valid_v=None,             # 可传入你的 valid_v 或 torch.isfinite(disp_gt_v)
                        save_npy=False,
                        save_v_grad=True,         # 额外导出 disp_pred_v 的梯度图
                        cmap='turbo'):
    """
    每个样本保存到 save_root/sample_xxxx：
      - left.png / right.png
      - disp_pred_h.png（直接保存）
      - disp_pred_v.png（robust 可视化，保证看出形状）
      - disp_gt_h.png / disp_gt_v.png（robust 可视化）
      - 可选：disp_pred_v_grad.png（梯度图，更显结构）
      - 可选：*.npy 原始数据
    """
    save_dir = os.path.join(save_root, f"sample_{idx:04d}")
    os.makedirs(save_dir, exist_ok=True)

    # RGB
    _save_rgb(os.path.join(save_dir, "left.png"),  left)
    _save_rgb(os.path.join(save_dir, "right.png"), right)

    # 视差（水平：直接保存；垂直与 GT：robust 可视化以保证有形状）
    _save_disp(os.path.join(save_dir, "disp_pred_h.png"), disp_pred_h, cmap=cmap, robust=True)
    _save_disp(os.path.join(save_dir, "disp_pred_v.png"), disp_pred_v, cmap=cmap, robust=True, valid_mask=valid_v)
    _save_disp(os.path.join(save_dir, "disp_pred.png"), disp_pred, cmap=cmap, robust=True)
    _save_disp(os.path.join(save_dir, "disp_gt_h.png"),   disp_gt_h,   cmap=cmap, robust=True, valid_mask=None)
    _save_disp(os.path.join(save_dir, "disp_gt_v.png"),   disp_gt_v,   cmap=cmap, robust=True, valid_mask=valid_v)
    _save_disp(os.path.join(save_dir, "disp_gt.png"), disp_gt, cmap=cmap, robust=True, valid_mask=None)

    if save_v_grad:
        _save_disp_grad(os.path.join(save_dir, "disp_pred_v_grad.png"), disp_pred_v)

    if save_npy:
        np.save(os.path.join(save_dir, "left.npy"),        _to_numpy(left))
        np.save(os.path.join(save_dir, "right.npy"),       _to_numpy(right))
        np.save(os.path.join(save_dir, "disp_pred_h.npy"), _to_numpy(disp_pred_h))
        np.save(os.path.join(save_dir, "disp_pred_v.npy"), _to_numpy(disp_pred_v))
        np.save(os.path.join(save_dir, "disp_gt_h.npy"),   _to_numpy(disp_gt_h))
        np.save(os.path.join(save_dir, "disp_gt_v.npy"),   _to_numpy(disp_gt_v))


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


def get_unpadded_H_inv_simple(H_inv_padded, pad_left, pad_top):
    """将填充后的单应矩阵变换回原始坐标系"""
    device = H_inv_padded.device
    dtype = H_inv_padded.dtype

    # 反向变换：填充坐标系 → 原始坐标系
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

    # H_recovered = T_pad_to_orig @ H_padded @ T_orig_to_pad
    return torch.bmm(torch.bmm(T_pad_to_orig, H_inv_padded), T_orig_to_pad)


class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ETH3D stereo validation:
       - bad1.0(all): %(|pred-gt| > 1.0) over valid
       - bad1.0(noc): %(|pred-gt| > 1.0) over valid ∧ (mask0nocc == 255)
       - RMSE(all):   sqrt(mean((pred-gt)^2)) over valid
    """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    save_root = "/home/kemove/zxc/training_EH3D/MonSter-main_2D_double/MonSter-main_2D_scale_autodl/picture/EH3D"
    save_dirs = {
        "pred_ecover": os.path.join(save_root, "pred_ecover"),
        "disp_pred": os.path.join(save_root, "disp_pred"),
        "disp_pred_vertical": os.path.join(save_root, "disp_pred_vertical"),
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    bad1_all_list, bad1_noc_list, rmse_all_list, elapsed_list = [], [], [], []

    def to_hw(x):
        """Squeeze到[H,W]（兼容 [1,1,H,W] / [1,H,W] / [H,W,1]）。"""
        if isinstance(x, torch.Tensor):
            if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0).squeeze(0)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x.squeeze(-1)
            return x.contiguous()
        elif isinstance(x, np.ndarray):
            if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1:
                x = x.squeeze(0).squeeze(0)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x.squeeze(-1)
            return x
        else:
            return x

    for val_id in range(len(val_dataset)):
        # 你的ETH3D返回格式（含valid）
        # (A,B,C,D,GT_files), left, right, disp_gt_h, valid_h, disp_gt_v, valid_v, disp_gt, valid, H_invs, H_inv_ls
        first_tuple, left, right, _, _, _, _, disp_gt, valid, H_invs, H_inv_ls = val_dataset[val_id]
        GT_files = first_tuple[-1] if isinstance(first_tuple, (list, tuple)) else first_tuple

        # ---- 推理 ----
        left  = left[None].cuda()
        right = right[None].cuda()
        H_inv_ls_t = torch.from_numpy(H_inv_ls).unsqueeze(0).cuda()
        H_invss = H_inv_ls_t

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            start = time.time()
            disp_pred_h, disp_pred_v, _ = model(left, right, H_inv_ls_t, iters=iters, test_mode=True)
            end = time.time()
        if val_id > 50:
            elapsed_list.append(end - start)

        # 去pad + 融合为最终水平视差
        disp_pred_h = padder.unpad(disp_pred_h)
        disp_pred_v = padder.unpad(disp_pred_v)
        pred_ecover = inverse_transform_verification1_fully_parallel(disp_pred_h, disp_pred_v, H_invss)

        # ---- 统一到 [H,W]，放同设备 ----
        device = left.device
        pred_ecover = to_hw(pred_ecover).to(device)
        disp_gt     = to_hw(disp_gt).to(device)
        valid       = to_hw(valid).to(device)

        # 基本一致性检查
        assert pred_ecover.shape == disp_gt.shape, (pred_ecover.shape, disp_gt.shape)
        assert valid.shape == disp_gt.shape, (valid.shape, disp_gt.shape)

        fname = f"{val_id}.png"
        vmax_vis = 192.0  # 与评估阈值一致，便于可视化对齐
        _save_disp_with_plt(
            pred_ecover.squeeze(0).detach().float().cpu().numpy(),
            os.path.join(save_dirs["pred_ecover"], fname), vmax=vmax_vis
        )
        _save_disp_with_plt(
            disp_pred_h.squeeze(0).squeeze(0).detach().float().cpu().numpy(),
            os.path.join(save_dirs["disp_pred"], fname), vmax=vmax_vis
        )
        _save_disp_with_plt(
            disp_pred_v.squeeze(0).squeeze(0).detach().float().cpu().numpy(),
            os.path.join(save_dirs["disp_pred_vertical"], fname), vmax=vmax_vis
        )

        # 读取NOC掩码（源代码口径：255=非遮挡）
        noc_path = GT_files.replace('disp0GT.pfm', 'mask0nocc.png')
        occ_np = np.array(Image.open(noc_path))  # uint8
        if occ_np.ndim == 3:
            occ_np = occ_np[..., 0]
        occ_mask = to_hw(torch.from_numpy(np.ascontiguousarray(occ_np))).to(device)
        assert occ_mask.shape == disp_gt.shape, (occ_mask.shape, disp_gt.shape)

        # ---- 源代码逻辑的掩码定义 ----
        # ALL：使用valid
        # NOC：使用 valid ∧ (mask0nocc == 255)
        valid_all_flat = (valid.flatten() >= 0.5)
        valid_noc_flat = valid_all_flat & (occ_mask.flatten() == 255)

        # ---- 误差与指标 ----
        # stereo: 绝对误差；bad1.0: |err| > 1.0
        err_abs = (pred_ecover - disp_gt).abs()
        err_flat = err_abs.flatten()

        if valid_all_flat.any():
            bad1_all = 100.0 * (err_flat[valid_all_flat] > 1.0).float().mean().item()
            rmse_all = torch.sqrt(((pred_ecover - disp_gt).flatten()[valid_all_flat] ** 2).mean()).item()
        else:
            bad1_all, rmse_all = float('nan'), float('nan')

        if valid_noc_flat.any():
            bad1_noc = 100.0 * (err_flat[valid_noc_flat] > 1.0).float().mean().item()
        else:
            bad1_noc = float('nan')

        # 可选：sanity打印，确保 all ⊃ noc
        if val_id < 3:
            print(f"[{val_id}] counts all/noc = {int(valid_all_flat.sum())}/{int(valid_noc_flat.sum())}  "
                  f"bad1_all={bad1_all:.4f}%  bad1_noc={bad1_noc:.4f}%")

        bad1_all_list.append(bad1_all)
        bad1_noc_list.append(bad1_noc)
        rmse_all_list.append(rmse_all)

    # ---- 汇总 ----
    bad1_all_mean = float(np.nanmean(np.array(bad1_all_list))) if bad1_all_list else float('nan')
    bad1_noc_mean = float(np.nanmean(np.array(bad1_noc_list))) if bad1_noc_list else float('nan')
    rmse_all_mean = float(np.nanmean(np.array(rmse_all_list))) if rmse_all_list else float('nan')

    if elapsed_list:
        avg_runtime = float(np.mean(elapsed_list))
        fps = 1.0 / avg_runtime if avg_runtime > 0 else float('inf')
    else:
        avg_runtime, fps = float('nan'), float('nan')

    print(f"Validation ETH3D: bad1.0(all) {bad1_all_mean:.4f}%, "
          f"bad1.0(noc) {bad1_noc_mean:.4f}%, RMSE(all) {rmse_all_mean:.4f}px, "
          f"{format(fps, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")

    return {
        'eth3d-bad1.0-all': bad1_all_mean,
        'eth3d-bad1.0-noc': bad1_noc_mean,
        'eth3d-rmse-all':   rmse_all_mean,
        'eth3d-fps':        fps,
    }


# def validate_eth3d(model, iters=32, mixed_prec=False):
#     """ Peform validation using the ETH3D (train) split """
#     model.eval()
#     aug_params = {}
#     val_dataset = datasets.ETH3D(aug_params)
#
#     epe_list_h = []
#     out_list_h = []
#     epe_list_v = []
#     out_list_v = []
#     epe_list = []
#     out_list = []
#
#     out_list, epe_list, elapsed_list = [], [], []
#     for val_id in range(len(val_dataset)):
#         # (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
#         (A,B,C,D,GT_files),left, right, disp_gt_h, valid_h, disp_gt_v, valid_v, disp_gt, valid, H_invs, H_inv_ls = val_dataset[val_id]
#
#         left = left[None].cuda()
#         right = right[None].cuda()
#         H_inv_ls = torch.from_numpy(H_inv_ls).unsqueeze(0).cuda()
#
#         H_invss = H_inv_ls
#         h_orig, w_orig = left.shape[-2:]
#
#         padder = InputPadder(left.shape, divis_by=32)
#         left, right = padder.pad(left, right)
#
#         h_padded, w_padded = left.shape[-2:]
#         pad_left = (w_padded - w_orig) // 2
#         pad_top = (h_padded - h_orig) // 2
#
#         H_inv_ls = get_padded_H_inv_simple(H_inv_ls, pad_left, pad_top)
#
#         with torch.no_grad():
#             disp_pred, disp_pred_vertical, pred_ecover = model(left, right, H_inv_ls, iters=iters,
#                                                                test_mode=True)
#         disp_pred = padder.unpad(disp_pred)
#         disp_pred_vertical = padder.unpad(disp_pred_vertical)  # 我猜测就是在这个地方尺寸变化引起的。
#         # pred_ecover =padder.unpad(pred_ecover)
#         # H_invss=get_unpadded_H_inv_simple(H_inv_ls, pad_left, pad_top)
#
#         pred_ecover = inverse_transform_verification1_fully_parallel(disp_pred, disp_pred_vertical, H_invss)
#
#         disp_pred1 = disp_pred.squeeze(0)
#         disp_pred_vertical1 = disp_pred_vertical.squeeze(0)
#         pred_ecover1 = pred_ecover.squeeze(0)
#         # disp_pred_vertical1_np = disp_pred_vertical1.squeeze().cpu().numpy()  # 去除第一维，转为numpy
#         # pd.DataFrame(disp_pred_vertical1_np).to_csv('disp_pred_vertical1.csv', index=False, header=False)
#         save_stereo_results(
#             save_root="/home/kemove/zxc/training_EH3D/MonSter-main_2D_double/MonSter-main_2D_scale/outputs/picture",
#             idx=val_id,
#             left=left.squeeze(0),  # [C,H,W]
#             right=right.squeeze(0),
#             disp_pred_h=disp_pred1,
#             disp_pred_v=disp_pred_vertical1,
#             disp_pred=pred_ecover1,
#             disp_gt_h=disp_gt_h,
#             disp_gt_v=disp_gt_v,
#             disp_gt=disp_gt,
#             valid_v=valid_v,
#             save_npy=False,  # 如果不需要 npy 保存可以改成 False
#             save_v_grad=True,
#             cmap='turbo'
#         )
#
#         pred_ecover = pred_ecover.squeeze(0).cuda()
#         disp_pred = disp_pred.squeeze(0).cuda()
#         disp_pred_vertical = disp_pred_vertical.squeeze(0).cuda()
#         disp_gt_h = disp_gt_h.cuda()
#         disp_gt_v = disp_gt_v.cuda()
#         disp_gt = disp_gt.cuda()
#         valid_h = valid_h.cuda()
#         valid_v = valid_v.cuda()
#         valid = valid.cuda()
#         # pred_ecover =padder.unpad(pred_ecover)
#
#         occ_mask = Image.open(GT_files.replace('disp0GT.pfm', 'mask0nocc.png'))
#
#         occ_mask = np.ascontiguousarray(occ_mask).flatten()
#
#
#
#         assert disp_pred.shape == disp_gt_h.shape, (disp_pred.shape, disp_gt_h.shape)
#         assert disp_pred_vertical.shape == disp_gt_v.shape, (disp_pred_vertical.shape, disp_gt_v.shape)
#         assert pred_ecover.shape == disp_gt.shape, (pred_ecover.shape, disp_gt.shape)
#         epe_h = torch.sum((disp_pred - disp_gt_h) ** 2, dim=0).sqrt()
#         epe_flattened_h = epe_h.flatten()
#         val_h = (valid_h.flatten() >= 0.5) & (disp_gt_h.abs().flatten() < 192)
#
#         out_h = (epe_flattened_h > 3.0)
#         image_out_h = out_h[val_h].float().mean().item()
#         image_epe_h = epe_flattened_h[val_h].mean().item()
#         epe_list_h.append(epe_flattened_h[val_h].mean().item())
#         out_list_h.append(out_h[val_h].cpu().numpy())
#
#         # 只有垂直的验证
#         epe_v = torch.sum((disp_pred_vertical - disp_gt_v) ** 2, dim=0).sqrt()
#         epe_flattened_v = epe_v.flatten()
#         val_v = (valid_v.flatten() >= 0.5) & (disp_gt_v.abs().flatten() < 192)
#
#         if val_v.sum() > 0:
#
#             out_v = (epe_flattened_v > 3.0)
#             image_out_v = out_v[val_v].float().mean().item()
#             image_epe_v = epe_flattened_v[val_v].mean().item()
#             epe_list_v.append(epe_flattened_v[val_v].mean().item())
#             out_list_v.append(out_v[val_v].cpu().numpy())
#         else:
#             image_out_v = 0.0
#             image_epe_v = 0.0
#             epe_list_v.append(0.0)
#             out_list_v.append(np.array([]))
#         # 纯水平视差的验证
#         epe = torch.sum((pred_ecover - disp_gt) ** 2, dim=0).sqrt()
#         epe_flattened = epe.flatten()
#         val = (valid.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)
#
#         out = (epe_flattened > 3.0)
#         image_out = out[val].float().mean().item()
#         image_epe = epe_flattened[val].mean().item()
#         epe_list.append(epe_flattened[val].mean().item())
#         out_list.append(out[val].cpu().numpy())
#
#     epe_list_h = np.array(epe_list_h)
#     out_list_h = np.concatenate(out_list_h)
#
#     epe_h = np.mean(epe_list_h)
#     d1_h = 100 * np.mean(out_list_h)
#
#     epe_list_v = np.array(epe_list_v)
#     out_list_v = np.concatenate(out_list_v)
#
#     epe_v = np.mean(epe_list_v)
#     d1_v = 100 * np.mean(out_list_v)
#
#     epe_list = np.array(epe_list)
#     out_list = np.concatenate(out_list)
#
#     epe = np.mean(epe_list)
#     d1 = 100 * np.mean(out_list)
#
#     os.makedirs("logs_txt", exist_ok=True)
#     open("logs/val_kitti5.txt", "a").write(
#         f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Step {1}: EPE {epe:.4f}, D1 {d1:.4f}, EPE_h {epe_h:.4f}, D1_h {d1_h:.4f}, EPE_v {epe_v:.4f}, D1_v {d1_v:.4f}\n")
#     print(
#         f"Validation KITTI: EPE {epe}, D1 {d1}, EPE_h {epe_h}, D1_h {d1_h}, EPE_v {epe_v}, D1_v {d1_v})")
#
#     return {'kitti-epe': epe, 'kitti-d1': d1}

    # out_list, epe_list = [], []
    # for val_id in range(len(val_dataset)):
    #     (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
    #     image1 = image1[None].cuda()
    #     image2 = image2[None].cuda()
    #
    #     padder = InputPadder(image1.shape, divis_by=32)
    #     image1, image2 = padder.pad(image1, image2)
    #     with torch.no_grad():
    #         with autocast(enabled=mixed_prec):
    #             flow_pr = model(image1, image2, iters=iters, test_mode=True)
    #
    #     flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
    #     assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
    #     epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
    #
    #     epe_flattened = epe.flatten()
    #
    #     occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
    #
    #     occ_mask = np.ascontiguousarray(occ_mask).flatten()
    #
    #     val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
    #     # val = (valid_gt.flatten() >= 0.5)
    #     out = (epe_flattened > 1.0)
    #     image_out = out[val].float().mean().item()
    #     image_epe = epe_flattened[val].mean().item()
    #     logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
    #     epe_list.append(image_epe)
    #     out_list.append(image_out)
    #
    # epe_list = np.array(epe_list)
    # out_list = np.array(out_list)
    #
    # epe = np.mean(epe_list)
    # d1 = 100 * np.mean(out_list)
    #
    # print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    # return {'eth3d-epe': epe, 'eth3d-d1': d1}

def _save_disp_with_plt(arr_np, out_path, vmax=192.0):
    """用 plt 保存单张视差图为 png。"""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.axis('off')
    plt.imshow(arr_np, cmap='jet')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    # aug_params = {'crop_size': list([540, 960])}
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    save_root = "/home/kemove/zxc/training_EH3D/MonSter-main_2D_double/MonSter-main_2D_scale_autodl/depth"
    save_dirs = {
        "pred_ecover": os.path.join(save_root, "pred_ecover"),
        "disp_pred": os.path.join(save_root, "disp_pred"),
        "disp_pred_vertical": os.path.join(save_root, "disp_pred_vertical"),
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)


    epe_list_h = []
    out_list_h = []
    epe_list_v = []
    out_list_v = []
    epe_list = []
    out_list = []

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        #(imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        (imageL_file,A,B,C,D), left, right, disp_gt_h, valid_h, disp_gt_v, valid_v, disp_gt, valid, H_invs, H_inv_ls =val_dataset[val_id]

        left = left[None].cuda()
        right = right[None].cuda()
        H_inv_ls=torch.from_numpy(H_inv_ls).unsqueeze(0).cuda()

        H_invss = H_inv_ls
        h_orig, w_orig = left.shape[-2:]

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        h_padded, w_padded = left.shape[-2:]
        pad_left = (w_padded - w_orig) // 2
        pad_top = (h_padded - h_orig) // 2

        H_inv_ls = get_padded_H_inv_simple(H_inv_ls, pad_left, pad_top)

        with torch.no_grad():
            disp_pred, disp_pred_vertical, pred_ecover = model(left, right, H_inv_ls, iters=iters,
                                                               test_mode=True)

        disp_pred = padder.unpad(disp_pred)
        disp_pred_vertical = padder.unpad(disp_pred_vertical)  # 我猜测就是在这个地方尺寸变化引起的。
        #pred_ecover =padder.unpad(pred_ecover)
        #H_invss=get_unpadded_H_inv_simple(H_inv_ls, pad_left, pad_top)

        pred_ecover = inverse_transform_verification1_fully_parallel(disp_pred, disp_pred_vertical, H_invss)

        pred_ecover= pred_ecover.squeeze(0).cuda()
        disp_pred = disp_pred.squeeze(0).cuda()
        disp_pred_vertical = disp_pred_vertical.squeeze(0).cuda()
        disp_gt_h= disp_gt_h.cuda()
        disp_gt_v= disp_gt_v.cuda()
        disp_gt= disp_gt.cuda()
        valid_h = valid_h.cuda()
        valid_v = valid_v.cuda()
        valid = valid.cuda()
        # pred_ecover =padder.unpad(pred_ecover)

        assert disp_pred.shape == disp_gt_h.shape, (disp_pred.shape, disp_gt_h.shape)
        assert disp_pred_vertical.shape == disp_gt_v.shape, (disp_pred_vertical.shape, disp_gt_v.shape)
        assert pred_ecover.shape == disp_gt.shape, (pred_ecover.shape, disp_gt.shape)


        # #这一部分代码只是为了生成逆向深度图的
        # pred_ecover1 = disp_pred.squeeze(0).detach().float().cpu().numpy()
        # pred_ecover1_max = pred_ecover1.max()
        # pred_ecover1 = pred_ecover1_max-pred_ecover1

        fname = f"{Path(imageL_file).stem}.csv"

        pred_np = pred_ecover.squeeze(0).detach().float().cpu().numpy()

        save_path = os.path.join(save_dirs["pred_ecover"], fname)
        np.savetxt(save_path, pred_np, delimiter=",")

        # fname =f"{Path(imageL_file).stem}.png"
        # vmax_vis = 192.0  # 与评估阈值一致，便于可视化对齐
        # _save_disp_with_plt(
        #     pred_ecover.squeeze(0).detach().float().cpu().numpy(),
        #     # pred_ecover1,
        #     os.path.join(save_dirs["pred_ecover"], fname), vmax=vmax_vis
        # )
        # _save_disp_with_plt(
        #     disp_pred.squeeze(0).detach().float().cpu().numpy(),
        #     os.path.join(save_dirs["disp_pred"], fname), vmax=vmax_vis
        # )
        # _save_disp_with_plt(
        #     disp_pred_vertical.squeeze(0).detach().float().cpu().numpy(),
        #     os.path.join(save_dirs["disp_pred_vertical"], fname), vmax=vmax_vis
        # )  #这个地方暂时先不进行保存

        epe_h = torch.sum((disp_pred - disp_gt_h) ** 2, dim=0).sqrt()
        epe_flattened_h = epe_h.flatten()
        val_h = (valid_h.flatten() >= 0.5) & (disp_gt_h.abs().flatten() < 192)

        out_h = (epe_flattened_h > 2.0)   #这个地方的评价指标是out-3.0还是out-2.0
        image_out_h = out_h[val_h].float().mean().item()
        image_epe_h = epe_flattened_h[val_h].mean().item()
        epe_list_h.append(epe_flattened_h[val_h].mean().item())
        out_list_h.append(out_h[val_h].cpu().numpy())

        # 只有垂直的验证
        epe_v = torch.sum((disp_pred_vertical - disp_gt_v) ** 2, dim=0).sqrt()
        epe_flattened_v = epe_v.flatten()
        val_v = (valid_v.flatten() >= 0.5) & (disp_gt_v.abs().flatten() < 192)

        if val_v.sum() > 0:

            out_v = (epe_flattened_v >2.0)
            image_out_v = out_v[val_v].float().mean().item()
            image_epe_v = epe_flattened_v[val_v].mean().item()
            epe_list_v.append(epe_flattened_v[val_v].mean().item())
            out_list_v.append(out_v[val_v].cpu().numpy())
        else:
            image_out_v = 0.0
            image_epe_v = 0.0
            epe_list_v.append(0.0)
            out_list_v.append(np.array([]))
        # 纯水平视差的验证
        epe = torch.sum((pred_ecover - disp_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        val = (valid.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list_h = np.array(epe_list_h)
    out_list_h = np.concatenate(out_list_h)

    epe_h = np.mean(epe_list_h)
    d1_h = 100 * np.mean(out_list_h)

    epe_list_v = np.array(epe_list_v)
    out_list_v = np.concatenate(out_list_v)

    epe_v = np.mean(epe_list_v)
    d1_v = 100 * np.mean(out_list_v)

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    os.makedirs("logs_txt", exist_ok=True)
    open("logs/val_kitti5.txt", "a").write(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Step {1}: EPE {epe:.4f}, D1 {d1:.4f}, EPE_h {epe_h:.4f}, D1_h {d1_h:.4f}, EPE_v {epe_v:.4f}, D1_v {d1_v:.4f}\n")
    print(
        f"Validation KITTI: EPE {epe}, D1 {d1}, EPE_h {epe_h}, D1_h {d1_h}, EPE_v {epe_v}, D1_v {d1_v})")

    return {'kitti-epe': epe, 'kitti-d1': d1}



# def validate_kitti(model, iters=32, mixed_prec=False, debug_first_k=3):
#     """KITTI 2015 (train) validation: D1-all / D1-bg / D1-fg with official rule.
#     Official rule: correct if |err| < 3px OR (|err|/|gt|) < 5%.
#     """
#     model.eval()
#     aug_params = {}
#     val_dataset = datasets.KITTI(aug_params, image_set='training')
#     torch.backends.cudnn.benchmark = True
#
#     d1_all_list, d1_bg_list, d1_fg_list = [], [], []
#
#     def compute_d1_official(d_pred, d_gt, valid_mask, obj_map):
#         # squeeze -> [H,W]
#         if d_pred.ndim == 3: d_pred = d_pred.squeeze(0)
#         if d_gt.ndim == 3: d_gt = d_gt.squeeze(0)
#         if valid_mask.ndim == 3: valid_mask = valid_mask.squeeze(0)
#         if obj_map.ndim == 3: obj_map = obj_map.squeeze(0)
#
#         valid_all = (valid_mask >= 0.5)  # 官方不限制 <192，如需保留可加: & (d_gt.abs() < 192)
#         bg_mask = (obj_map == 0)
#         fg_mask = ~bg_mask
#
#         err = (d_pred - d_gt).abs()
#         gt_abs = d_gt.abs().clamp_min(1e-5)
#
#         # 官方口径：正确是 "<" 门槛；错误为其补集 ⇒ (err >= 3) AND (rel >= 5%)
#         correct = (err < 3.0) | ((err / gt_abs) < 0.05)
#         bad = ~correct
#
#         n_all = int(valid_all.sum().item())
#         n_bg  = int((valid_all & bg_mask).sum().item())
#         n_fg  = int((valid_all & fg_mask).sum().item())
#
#         d1_all = 100.0 * bad[valid_all].float().mean().item() if n_all > 0 else float('nan')
#         d1_bg  = 100.0 * bad[valid_all & bg_mask].float().mean().item() if n_bg  > 0 else float('nan')
#         d1_fg  = 100.0 * bad[valid_all & fg_mask].float().mean().item() if n_fg  > 0 else float('nan')
#
#         return d1_all, d1_bg, d1_fg, n_all, n_bg, n_fg
#
#     # 结果日志目录
#     os.makedirs("logs", exist_ok=True)
#     os.makedirs("logs_txt", exist_ok=True)
#
#     for i in range(len(val_dataset)):
#         (_, left, right,
#          disp_gt_h, valid_h,
#          disp_gt_v, valid_v,
#          disp_gt, valid,
#          H_invs, H_inv_ls,
#          obj_map) = val_dataset[i]
#
#         # 前处理 & 推理（与你原流程一致）
#         left = left[None].cuda()
#         right = right[None].cuda()
#         H_inv_ls = torch.from_numpy(H_inv_ls).unsqueeze(0).cuda()
#         H_invss = H_inv_ls
#
#         h_orig, w_orig = left.shape[-2:]
#         padder = InputPadder(left.shape, divis_by=32)
#         left, right = padder.pad(left, right)
#
#         h_padded, w_padded = left.shape[-2:]
#         pad_left = (w_padded - w_orig) // 2
#         pad_top  = (h_padded - h_orig) // 2
#         H_inv_ls = get_padded_H_inv_simple(H_inv_ls, pad_left, pad_top)
#
#         with torch.no_grad():
#             disp_pred, disp_pred_vertical, pred_ecover = model(
#                 left, right, H_inv_ls, iters=iters, test_mode=True
#             )
#
#         # 复原到原尺寸，并得到最终水平视差预测
#         disp_pred = padder.unpad(disp_pred)
#         disp_pred_vertical = padder.unpad(disp_pred_vertical)
#         pred_ecover = inverse_transform_verification1_fully_parallel(
#             disp_pred, disp_pred_vertical, H_invss
#         ).squeeze(0).cuda()
#
#         disp_gt = disp_gt.cuda()
#         valid   = valid.cuda()
#         obj_map = obj_map.cuda()
#
#         # 基本一致性检查
#         assert pred_ecover.shape == disp_gt.shape, (pred_ecover.shape, disp_gt.shape)
#         assert obj_map.shape[-2:] == disp_gt.shape[-2:], (obj_map.shape, disp_gt.shape)
#
#         # 尺度自检（常见问题：GT或预测未 /256）
#         with torch.no_grad():
#             gt_max = float(disp_gt.max().item())
#             pr_max = float(pred_ecover.max().item())
#             if gt_max > 2048 or pr_max > 2048:
#                 print(f"[warn][{i}] suspicious disparity scale (max gt={gt_max:.1f}, pred={pr_max:.1f}). "
#                       f"Did you divide KITTI PNG disparities by 256?")
#
#         # 计算 D1 指标
#         d1_all, d1_bg, d1_fg, n_all, n_bg, n_fg = compute_d1_official(
#             pred_ecover, disp_gt, valid, obj_map
#         )
#
#         # 一致性自检（前 debug_first_k 张打印）
#         if i < debug_first_k and n_all > 0:
#             d1_from_parts = (d1_bg * n_bg + d1_fg * n_fg) / max(n_all, 1)
#             print(f"[{i}] D1_all={d1_all:.4f}% | D1_bg={d1_bg:.4f}% | D1_fg={d1_fg:.4f}% | "
#                   f"counts all/bg/fg={n_all}/{n_bg}/{n_fg} | "
#                   f"weighted={d1_from_parts:.4f}% (Δ={abs(d1_all - d1_from_parts):.4e}) | "
#                   f"obj_map_unique={torch.unique(obj_map).cpu().tolist()[:10]}")
#
#         d1_all_list.append(d1_all)
#         d1_bg_list.append(d1_bg)
#         d1_fg_list.append(d1_fg)
#
#     # 汇总
#     d1_all_mean = float(np.nanmean(np.array(d1_all_list))) if d1_all_list else float('nan')
#     d1_bg_mean  = float(np.nanmean(np.array(d1_bg_list)))  if d1_bg_list  else float('nan')
#     d1_fg_mean  = float(np.nanmean(np.array(d1_fg_list)))  if d1_fg_list  else float('nan')
#
#     with open("logs/val_kitti5.txt", "a") as f:
#         f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] D1-all {d1_all_mean:.4f}, D1-bg {d1_bg_mean:.4f}, D1-fg {d1_fg_mean:.4f}\n")
#
#     print(f"Validation KITTI Stereo 2015 (official rule): "
#           f"D1-all={d1_all_mean:.4f}%, D1-bg={d1_bg_mean:.4f}%, D1-fg={d1_fg_mean:.4f}%")
#
#     return {
#         'kitti-d1-all': d1_all_mean,
#         'kitti-d1-bg':  d1_bg_mean,
#         'kitti-d1-fg':  d1_fg_mean,
#     }





    #     image1 = image1[None].cuda()
    #     image2 = image2[None].cuda()
    #
    #     padder = InputPadder(image1.shape, divis_by=32)
    #     image1, image2 = padder.pad(image1, image2)
    #
    #     with torch.no_grad():
    #         with autocast(enabled=mixed_prec):
    #             start = time.time()
    #             flow_pr = model(image1, image2, iters=iters, test_mode=True)
    #             end = time.time()
    #
    #     if val_id > 50:
    #         elapsed_list.append(end-start)
    #     flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
    #
    #     assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
    #     epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
    #
    #     epe_flattened = epe.flatten()
    #     val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
    #     # val = valid_gt.flatten() >= 0.5
    #
    #     out = (epe_flattened > 3.0)
    #     image_out = out[val].float().mean().item()
    #     image_epe = epe_flattened[val].mean().item()
    #     if val_id < 9 or (val_id+1)%10 == 0:
    #         logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
    #     epe_list.append(epe_flattened[val].mean().item())
    #     out_list.append(out[val].cpu().numpy())
    #
    #     # if val_id > 20:
    #     #     break
    #
    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)
    #
    # epe = np.mean(epe_list)
    # d1 = 100 * np.mean(out_list)
    #
    # avg_runtime = np.mean(elapsed_list)
    #
    # print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    # return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_vkitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the vkitti (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.VKITTI2(aug_params)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)   #这个验证的三个像素
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"VKITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # if val_id > 20:
        #     break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation VKITTI: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'vkitti-epe': epe, 'vkitti-d1': d1}



@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, disp_gt_h, valid_h, disp_gt_v, valid_v, flow_gt, valid_gt, H_invs, H_inv_ls = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        H_inv_ls=torch.from_numpy(H_inv_ls).unsqueeze(0).cuda()

        H_invss = H_inv_ls
        h_orig, w_orig = image1.shape[-2:]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        h_padded, w_padded = image1.shape[-2:]
        pad_left = (w_padded - w_orig) // 2
        pad_top = (h_padded - h_orig) // 2

        H_inv_ls = get_padded_H_inv_simple(H_inv_ls, pad_left, pad_top)

        with autocast(enabled=mixed_prec):
            start = time.time()
            disp_pred,disp_pred_vertical,pred_ecover = model(image1, image2, H_inv_ls,iters=iters, test_mode=True)
            end = time.time()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        if val_id > 50:
            elapsed_list.append(end-start)

        disp_pred = padder.unpad(disp_pred)
        disp_pred_vertical = padder.unpad(disp_pred_vertical)  # 我猜测就是在这个地方尺寸变化引起的。
        # pred_ecover =padder.unpad(pred_ecover)
        # H_invss=get_unpadded_H_inv_simple(H_inv_ls, pad_left, pad_top)

        flow_pr = inverse_transform_verification1_fully_parallel(disp_pred, disp_pred_vertical, H_invss)

        device =flow_pr.device
        flow_gt = flow_gt.to(device)
        valid_gt = valid_gt.to(device)

        flow_pr = flow_pr.squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe = torch.abs(flow_pr - flow_gt)

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if(np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"Scene Flow Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")

        print('epe', epe[val].mean().item())
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)
    # f = open('test.txt', 'a')
    # f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    print(f"Validation Scene Flow: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)" )
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_driving(model, iters=32, mixed_prec=False):
    """ Peform validation using the DrivingStereo (test) split """
    model.eval()
    aug_params = {}
    # val_dataset = datasets.DrivingStereo(aug_params, image_set='test')
    val_dataset = datasets.DrivingStereo(aug_params, image_set='cloudy')
    print(len(val_dataset))
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    out1_list, out2_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.autocast(device_type='cuda', enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        out1 = (epe_flattened > 1.0)
        out2 = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"Driving Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        out1_list.append(out1[val].cpu().numpy())
        out2_list.append(out2[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    out1_list = np.concatenate(out1_list)
    out2_list = np.concatenate(out2_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)
    bad_2 = 100 * np.mean(out2_list)
    bad_1 = 100 * np.mean(out1_list)
    avg_runtime = np.mean(elapsed_list)

    print(f"Validation DrivingStereo: EPE {epe}, bad1 {bad_1}, bad2 {bad_2}, bad3 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'driving-epe': epe, 'driving-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        a = input('input something')
        print(a)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default="/home/kemove/zxc/training_EH3D/MonSter-main_2D_double/MonSter-main_2D_scale_autodl/pretrained/kitti_large/100000.pth")   #画三维点云用的训练权重
    parser.add_argument('--dataset', help="dataset for evaluation", default='kitti', choices=["eth3d", "kitti", "sceneflow", "vkitti", "driving"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total number of parameters: {total_params:.2f}M")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of trainable parameters: {trainable_params:.2f}M")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        logging.info(args.restore_ckpt)
        assert os.path.exists(args.restore_ckpt)
        checkpoint = torch.load(args.restore_ckpt)
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            # ckpt['module.' + key] = checkpoint[key]
            if key.startswith("module."):
                ckpt[key] = checkpoint[key]  # 保持原样
            else:
                ckpt["module." + key] = checkpoint[key]  # 添加 "module."

        model.load_state_dict(ckpt, strict=True)

        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'vkitti':
        validate_vkitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'driving':
        validate_driving(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
