import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2
from core.geometry import Combined_Geo_Encoding_Volume,Combined_Geo_Encoding_Volume_Vertical,Combined_Geo_Encoding_Volume_2D_Reduction
from core.Combined_Geo_Encoding_Volume_2D_Reduction import Combined_Geo_Encoding_Volume_2D_Reduction_DepthEnhanced
from core.submodule import *
from core.gwc import *
from core.refinement import REMP, REMP_Vertical
from core.warp import disp_warp,disp_warp_vertical
import matplotlib.pyplot as plt
import math
try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
import sys
sys.path.append('./Depth-Anything-V2-list3')
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder

import torch.nn.functional as F

def pad_cost_volume_D_right(vol, target_D):
    """
    vol: [B, G, D, H, W]
    target_D: 统一/对齐后的 D（建议是 8 的倍数）
    return:
      vol_padded: [B, G, target_D, H, W]
      D_orig: 原始 D（用于后续裁剪）
    """
    B, G, D, H, W = vol.shape
    if D == target_D:
        return vol.contiguous(), D
    pad_D = target_D - D
    # F.pad: (Wl,Wr, Hl,Hr, Dl,Dr, Cl,Cr, Bl,Br)
    vol_padded = F.pad(vol, (0, 0, 0, 0, 0, pad_D))
    return vol_padded.contiguous(), D

def crop_cost_volume_D_right(vol, D_orig):
    """
    vol: [B, C, D, H, W]  或  [B, D, H, W]（分类输出）
    将 D 维裁回到 D_orig
    """
    if vol.dim() == 5:
        return vol[:, :, :D_orig, ...].contiguous()
    elif vol.dim() == 4:
        return vol[:, :D_orig, ...].contiguous()
    else:
        raise ValueError("vol must be 4D or 5D tensor")


class EnhancedLocalConsistencyAlignment:
    """
    简化增强版局部一致性对齐方法
    主要针对严重未对齐系统（大垂直视差）进行优化
    在基础版本上增加：
    1. 自动垂直视差检测
    2. 自适应处理策略
    3. 简化的置信度评估
    """

    def __init__(self, patch_size=16, overlap=0.5, verbose=False):
        self.patch_size = patch_size
        self.overlap = overlap
        self.verbose = verbose  # 控制详细输出
        self.patch_results = []  # 存储每个patch的处理结果

        # 垂直视差处理的阈值设定
        self.large_vertical_threshold = 0.4  # 垂直/水平比例超过40%认为是大垂直视差
        self.correlation_threshold_low = 0.2  # 弱相关性阈值（针对大垂直视差调整）
        self.correlation_threshold_high = 0.5  # 强相关性阈值

    def detect_vertical_disparity_level(self, disp_h, disp_v):
        """
        检测垂直视差水平

        Args:
            disp_h: 水平视差 (B, C, H, W) 或 (H, W)
            disp_v: 垂直视差 (B, C, H, W) 或 (H, W)

        Returns:
            dict: 包含垂直视差分析结果的字典
        """
        # 提取2D数据
        if disp_h.dim() == 3:
            disp_h_2d = disp_h[0]
            disp_v_2d = disp_v[0]
        else:
            disp_h_2d = disp_h
            disp_v_2d = disp_v

        # 计算有效区域的统计信息
        total_mag = torch.sqrt(disp_h_2d ** 2 + disp_v_2d ** 2)
        valid_mask = total_mag > 1e-3

        if valid_mask.sum() == 0:
            return {'is_large_vertical': False, 'vertical_ratio': 0.0, 'quality': 'unknown'}

        # 计算垂直和水平视差的平均幅度
        h_mag_mean = torch.abs(disp_h_2d[valid_mask]).mean().item()
        v_mag_mean = torch.abs(disp_v_2d[valid_mask]).mean().item()

        # 垂直/水平比例
        vertical_ratio = v_mag_mean / (h_mag_mean + 1e-6)

        # 判断对齐质量
        is_large_vertical = vertical_ratio > self.large_vertical_threshold

        if vertical_ratio < 0.1:
            quality = 'excellent'  # 很好的对齐
        elif vertical_ratio < 0.3:
            quality = 'good'  # 良好对齐
        elif vertical_ratio < 0.6:
            quality = 'poor'  # 较差对齐
        else:
            quality = 'very_poor'  # 很差对齐

        return {
            'is_large_vertical': is_large_vertical,
            'vertical_ratio': vertical_ratio,
            'quality': quality,
            'h_mag_mean': h_mag_mean,
            'v_mag_mean': v_mag_mean
        }

    def align_with_adaptation(self, depth_mono, disp_h, disp_v):
        """
        自适应对齐方法：根据垂直视差大小选择处理策略

        Args:
            depth_mono: (B, C, H, W) 单目深度
            disp_h: (B, C, H, W) 水平视差
            disp_v: (B, C, H, W) 垂直视差

        Returns:
            aligned_h: (B, C, H, W) 对齐后的水平视差
            aligned_v: (B, C, H, W) 对齐后的垂直视差
        """
        # 检查输入维度
        assert depth_mono.dim() == 3, f"期望4维tensor，得到{depth_mono.dim()}维"
        assert depth_mono.shape == disp_h.shape == disp_v.shape, "输入tensor尺寸必须相同"

        # 检测垂直视差水平
        analysis = self.detect_vertical_disparity_level(disp_h, disp_v)

        if self.verbose:
            print(f"垂直视差分析: 比例={analysis['vertical_ratio']:.3f}, 质量={analysis['quality']}")
            if analysis['is_large_vertical']:
                print("检测到大垂直视差，启用增强处理模式")

        # 根据垂直视差情况选择处理策略
        if analysis['is_large_vertical']:
            return self._align_with_large_vertical_handling(depth_mono, disp_h, disp_v, analysis)
        else:
            return self._align_standard(depth_mono, disp_h, disp_v, analysis)

    def _align_with_large_vertical_handling(self, depth_mono, disp_h, disp_v, analysis):
        """
        针对大垂直视差的特殊处理
        主要改进：
        1. 调整相关性阈值
        2. 增加置信度评估
        3. 更严格的有效性检查
        """
        B, H, W = depth_mono.shape

        # 提取2维数据进行处理
        depth_2d = depth_mono[0]
        disp_h_2d = disp_h[0]
        disp_v_2d = disp_v[0]
        h, w = depth_2d.shape

        # 初始化
        aligned_h_2d = torch.zeros_like(disp_h_2d)
        aligned_v_2d = torch.zeros_like(disp_v_2d)
        weight_sum = torch.zeros_like(disp_h_2d)
        confidence_map = torch.zeros_like(disp_h_2d)  # 置信度图

        stride = int(self.patch_size * (1 - self.overlap))
        patch_count = 0
        successful_patches = 0

        # 滑动窗口处理
        for i in range(0, h - self.patch_size + 1, stride):
            for j in range(0, w - self.patch_size + 1, stride):
                patch_count += 1

                # 提取局部patch
                depth_patch = depth_2d[i:i + self.patch_size, j:j + self.patch_size]
                disp_h_patch = disp_h_2d[i:i + self.patch_size, j:j + self.patch_size]
                disp_v_patch = disp_v_2d[i:i + self.patch_size, j:j + self.patch_size]

                # 使用增强的patch对齐方法
                result = self._align_patch_enhanced(
                    depth_patch, disp_h_patch, disp_v_patch,
                    patch_id=patch_count, position=(i, j), global_analysis=analysis
                )

                if result is None:
                    continue

                aligned_h_patch, aligned_v_patch, confidence, patch_info = result
                self.patch_results.append(patch_info)
                successful_patches += 1

                # 结合置信度的权重
                spatial_weight = self._gaussian_weight(self.patch_size,device=depth_2d.device, dtype=depth_2d.dtype)
                total_weight = spatial_weight * confidence

                # 累加结果
                aligned_h_2d[i:i + self.patch_size, j:j + self.patch_size] += aligned_h_patch * total_weight
                aligned_v_2d[i:i + self.patch_size, j:j + self.patch_size] += aligned_v_patch * total_weight
                weight_sum[i:i + self.patch_size, j:j + self.patch_size] += total_weight

                # 更新置信度图
                confidence_map[i:i + self.patch_size, j:j + self.patch_size] = torch.max(
                    confidence_map[i:i + self.patch_size, j:j + self.patch_size],
                    confidence * spatial_weight
                )

        # 归一化
        mask = weight_sum > 1e-6
        aligned_h_2d[mask] = aligned_h_2d[mask] / weight_sum[mask]
        aligned_v_2d[mask] = aligned_v_2d[mask] / weight_sum[mask]

        # 转换回4维tensor格式
        aligned_h = torch.zeros_like(disp_h)
        aligned_v = torch.zeros_like(disp_v)
        for b in range(B):
            aligned_h[b] = aligned_h_2d
            aligned_v[b] = aligned_v_2d

        if self.verbose:
            avg_confidence = confidence_map[mask].mean().item() if mask.sum() > 0 else 0
            print(
                f"大垂直视差处理完成: {successful_patches}/{patch_count} patches成功, 平均置信度: {avg_confidence:.3f}")

        return aligned_h, aligned_v

    def _align_standard(self, depth_mono, disp_h, disp_v, analysis):
        """
        标准对齐处理（针对垂直视差较小的情况）
        基本保持原始算法逻辑
        """
        B, H, W = depth_mono.shape

        # 提取2维数据进行处理
        depth_2d = depth_mono[0]
        disp_h_2d = disp_h[0]
        disp_v_2d = disp_v[0]
        h, w = depth_2d.shape

        # 初始化
        aligned_h_2d = torch.zeros_like(disp_h_2d)
        aligned_v_2d = torch.zeros_like(disp_v_2d)
        weight_sum = torch.zeros_like(disp_h_2d)

        stride = int(self.patch_size * (1 - self.overlap))
        patch_count = 0
        successful_patches = 0

        # 滑动窗口处理
        for i in range(0, h - self.patch_size + 1, stride):
            for j in range(0, w - self.patch_size + 1, stride):
                patch_count += 1

                # 提取局部patch
                depth_patch = depth_2d[i:i + self.patch_size, j:j + self.patch_size]
                disp_h_patch = disp_h_2d[i:i + self.patch_size, j:j + self.patch_size]
                disp_v_patch = disp_v_2d[i:i + self.patch_size, j:j + self.patch_size]

                # 使用标准patch对齐方法（基于原始版本）
                result = self._align_patch_standard(
                    depth_patch, disp_h_patch, disp_v_patch,
                    patch_id=patch_count, position=(i, j)
                )

                if result is None:
                    continue

                aligned_h_patch, aligned_v_patch, patch_info = result
                self.patch_results.append(patch_info)
                successful_patches += 1

                # 高斯权重
                weight = self._gaussian_weight(self.patch_size,device=depth_2d.device, dtype=depth_2d.dtype)

                # 累加结果
                aligned_h_2d[i:i + self.patch_size, j:j + self.patch_size] += aligned_h_patch * weight
                aligned_v_2d[i:i + self.patch_size, j:j + self.patch_size] += aligned_v_patch * weight
                weight_sum[i:i + self.patch_size, j:j + self.patch_size] += weight

        # 归一化
        mask = weight_sum > 0
        aligned_h_2d[mask] = aligned_h_2d[mask] / weight_sum[mask]
        aligned_v_2d[mask] = aligned_v_2d[mask] / weight_sum[mask]

        # 转换回4维tensor格式
        aligned_h = torch.zeros_like(disp_h)
        aligned_v = torch.zeros_like(disp_v)
        for b in range(B):
            aligned_h[b] = aligned_h_2d
            aligned_v[b] = aligned_v_2d

        if self.verbose:
            print(f"标准处理完成: {successful_patches}/{patch_count} patches成功")

        return aligned_h, aligned_v

    def _align_patch_enhanced(self, depth_patch, disp_h_patch, disp_v_patch,
                              patch_id, position, global_analysis):
        """
        增强的patch对齐方法，针对大垂直视差优化

        主要改进：
        1. 更严格的有效性检查
        2. 自适应相关性阈值
        3. 置信度评估
        4. 局部垂直视差一致性检查
        """
        # 计算视差幅度
        disp_mag_patch = torch.sqrt(disp_h_patch ** 2 + disp_v_patch ** 2 + 1e-6)

        # 更严格的有效掩码（针对大垂直视差情况）
        depth_mask = depth_patch > 0.1
        disp_mask = disp_mag_patch > 1e-3
        combined_mask = depth_mask & disp_mask
        valid_points = combined_mask.sum().item()

        # 对于大垂直视差，需要更多的有效点
        min_valid_points = max(15, int(self.patch_size * self.patch_size * 0.4))
        if valid_points < min_valid_points:
            return None

        # 提取有效数据
        depth_valid = depth_patch[combined_mask].flatten()
        disp_mag_valid = disp_mag_patch[combined_mask].flatten()
        disp_h_valid = disp_h_patch[combined_mask].flatten()
        disp_v_valid = disp_v_patch[combined_mask].flatten()

        # 统计信息
        depth_mean = depth_valid.mean().item()
        depth_std = depth_valid.std().item()
        disp_mean = disp_mag_valid.mean().item()
        disp_std = disp_mag_valid.std().item()

        # 检查局部垂直视差比例
        local_h_mag = torch.abs(disp_h_valid).mean().item()
        local_v_mag = torch.abs(disp_v_valid).mean().item()
        local_vertical_ratio = local_v_mag / (local_h_mag + 1e-6)

        # 局部归一化
        depth_norm = (depth_valid - depth_mean) / (depth_std + 1e-6)
        disp_norm = (disp_mag_valid - disp_mean) / (disp_std + 1e-6)

        # 计算相关性
        correlation = -(depth_norm * disp_norm).mean().item()

        # 置信度评估（简化版）
        confidence_factors = [
            min(1.0, abs(correlation) * 2),  # 相关性强度
            min(1.0, valid_points / (self.patch_size ** 2)),  # 有效点比例
            max(0.3, 1.0 - abs(local_vertical_ratio - global_analysis['vertical_ratio'])),  # 局部一致性
        ]
        confidence = np.prod(confidence_factors)

        # 自适应尺度计算（针对大垂直视差调整阈值）
        if abs(correlation) < self.correlation_threshold_low:
            # 相关性很弱，使用保守策略
            scale = disp_mean / (depth_mean + 1e-6)
            transform_type = "conservative"
        elif abs(correlation) < self.correlation_threshold_high:
            # 中等相关性，混合策略
            scale1 = disp_mean / (depth_mean + 1e-6)
            scale2 = -disp_std / (depth_std + 1e-6) * correlation
            scale = 0.6 * scale1 + 0.4 * scale2  # 更保守的混合
            transform_type = "mixed"
        else:
            # 强相关性，标准策略
            scale = -disp_std / (depth_std + 1e-6) * correlation
            transform_type = "standard"

        # 应用变换
        disp_mag_aligned = disp_mean + scale * (depth_patch - depth_mean)

        # 保持原始方向（关键：完整保持垂直视差信息）
        direction_h = disp_h_patch / (disp_mag_patch + 1e-6)
        direction_v = disp_v_patch / (disp_mag_patch + 1e-6)

        aligned_h = disp_mag_aligned * direction_h
        aligned_v = disp_mag_aligned * direction_v

        # 记录详细信息
        patch_info = {
            'patch_id': patch_id,
            'position': position,
            'valid_points': valid_points,
            'correlation': correlation,
            'confidence': confidence,
            'local_vertical_ratio': local_vertical_ratio,
            'scale': scale,
            'transform_type': transform_type
        }

        return aligned_h, aligned_v, confidence, patch_info

    def _align_patch_standard(self, depth_patch, disp_h_patch, disp_v_patch,
                              patch_id, position):
        """
        标准patch对齐方法（基于原始版本）
        """
        # 计算视差幅度
        disp_mag_patch = torch.sqrt(disp_h_patch ** 2 + disp_v_patch ** 2 + 1e-6)

        # 有效掩码
        mask = (depth_patch > 0.1) & (disp_mag_patch > 1e-3)
        valid_points = mask.sum().item()

        if valid_points < 10:
            return None

        # 提取有效数据
        depth_valid = depth_patch[mask].flatten()
        disp_mag_valid = disp_mag_patch[mask].flatten()

        # 记录原始统计信息
        depth_mean = depth_valid.mean().item()
        depth_std = depth_valid.std().item()
        disp_mean = disp_mag_valid.mean().item()
        disp_std = disp_mag_valid.std().item()

        # 局部归一化
        depth_norm = (depth_valid - depth_mean) / (depth_std + 1e-6)
        disp_norm = (disp_mag_valid - disp_mean) / (disp_std + 1e-6)

        # 计算相关性（深度和视差是反比关系，所以是负相关）
        correlation = -(depth_norm * disp_norm).mean().item()

        # 基于相关性强度决定变换（原始版本的逻辑）
        if abs(correlation) < 0.3:
            # 相关性弱，使用简单的均值比例
            scale = disp_mean / (depth_mean + 1e-6)
            transform_type = "weak_correlation"
        else:
            # 相关性强，使用基于标准差的尺度
            scale = -disp_std / (depth_std + 1e-6) * correlation
            transform_type = "strong_correlation"

        # 应用变换
        disp_mag_aligned = disp_mean + scale * (depth_patch - depth_mean)

        # 保持原始方向
        direction_h = disp_h_patch / (disp_mag_patch + 1e-6)
        direction_v = disp_v_patch / (disp_mag_patch + 1e-6)

        aligned_h = disp_mag_aligned * direction_h
        aligned_v = disp_mag_aligned * direction_v

        # 记录详细信息
        patch_info = {
            'patch_id': patch_id,
            'position': position,
            'valid_points': valid_points,
            'correlation': correlation,
            'scale': scale,
            'transform_type': transform_type
        }

        return aligned_h, aligned_v, patch_info

    def _gaussian_weight(self, size, sigma=0.3,device=None, dtype=None):
        """生成高斯权重"""
        x = torch.linspace(-1, 1, size,device=device, dtype=dtype)
        y = torch.linspace(-1, 1, size,device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        weight = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return weight

def align_with_enhanced_method(depth_mono, disp_h, disp_v, patch_size=16, overlap=0.5, verbose=False):
    """
    简化的增强对齐函数

    Args:
        depth_mono: (1, 1, 80, 184) 单目深度
        disp_h: (1, 1, 80, 184) 水平视差
        disp_v: (1, 1, 80, 184) 垂直视差
        patch_size: patch大小
        overlap: 重叠率
        verbose: 是否详细输出

    Returns:
        aligned_h: (1, 1, 80, 184) 对齐后的水平视差
        aligned_v: (1, 1, 80, 184) 对齐后的垂直视差
    """
    # 检查输入格式
    assert depth_mono.shape == disp_h.shape == disp_v.shape, "输入tensor尺寸必须相同"

    # 创建增强对齐器
    aligner = EnhancedLocalConsistencyAlignment(
        patch_size=patch_size,
        overlap=overlap,
        verbose=verbose
    )

    # 执行自适应对齐
    aligned_h, aligned_v = aligner.align_with_adaptation(depth_mono, disp_h, disp_v)

    return aligned_h, aligned_v



@torch.no_grad()
def inverse_transform_verification1_vectorized(
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



def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift


@torch.no_grad()
def fit_vertical_disp_from_invdepth_columnwise(
    monodepth: torch.Tensor,   # [H,W]  单目深度(或其缩放版)
    gt_disp_v: torch.Tensor,   # [H,W]  预测/引导的垂直视差(可为负)
    valid_mask: torch.Tensor = None,  # [H,W] 可选
    *,
    iters: int = 3,            # IRLS 迭代次数（2-4次足够）
    huber_delta: float = 1.0,  # Huber 阈值
    ridge: float = 1e-3,       # 岭回归稳定项
    min_points: int = 24,      # 每列最少有效点；不够则列回退
    allow_shift: bool = True,  # 是否拟合偏置 t(w)
    eps: float = 1e-6,
):
    """
    在每一列 w 上稳健拟合：gt_disp_v(h,w) ≈ s(w) * (1/monodepth(h,w)) + t(w)
    返回:
        disp_v_fit: [H,W]  拟合后的垂直视差（用于替换/校正 disp_mono_4x_vertical 的值）
        s: [W]             每列比例
        t: [W]             每列偏置（allow_shift=False 时恒为 0）
    """
    H, W = monodepth.shape
    device = monodepth.device
    dtype  = monodepth.dtype

    # x = 1/Z
    x = 1.0 / monodepth.clamp_min(eps)
    y = gt_disp_v

    # 有效掩码：允许负视差；只排除非数、极小深度/逆深度
    if valid_mask is None:
        valid_mask = torch.isfinite(x) & torch.isfinite(y) & (x > 0) & (y.abs() > 0)

    # 把每列的加权最小二乘写成向量化的 2x2 正规方程
    def solve_weighted_ls(x, y, w, allow_shift):
        # x,y,w: [H,W]
        wx   = w * x
        Sx2  = (wx * x).sum(dim=0) + ridge          # [W]
        Sx   = wx.sum(dim=0)                        # [W]
        S1   = w.sum(dim=0) + (ridge if allow_shift else 0.0)  # [W]
        Sxy  = (wx * y).sum(dim=0)                  # [W]
        Sy   = (w  * y).sum(dim=0)                  # [W]

        if allow_shift:
            # 解每列的 2x2：[[Sx2, Sx],[Sx, S1]] [s,t]^T = [Sxy, Sy]^T
            det = Sx2 * S1 - Sx * Sx
            det = torch.where(det.abs() < 1e-8, det.sign() * 1e-8, det)
            s = (Sxy * S1 - Sx * Sy) / det
            t = (Sx2 * Sy - Sx * Sxy) / det
        else:
            # 仅比例：最小二乘 s = (x^T W y) / (x^T W x)
            denom = Sx2
            denom = torch.where(denom.abs() < 1e-8, denom.sign() * 1e-8, denom)
            s = Sxy / denom
            t = torch.zeros_like(s)

        return s, t  # [W], [W]

    # 初始权重：有效点为 1，其他为 0
    w = valid_mask.to(dtype)

    # IRLS（Huber）
    s, t = solve_weighted_ls(x, y, w, allow_shift=allow_shift)
    for _ in range(iters):
        # 预测与残差
        y_hat = s.view(1, W) * x + t.view(1, W)      # [H,W]
        r = y - y_hat
        # Huber 权重（稳健）
        abs_r = r.abs()
        w = torch.where(
            abs_r <= huber_delta,
            torch.ones_like(abs_r),
            (huber_delta / (abs_r + eps))
        )
        # 仍然屏蔽无效像素
        w = w * valid_mask.to(dtype)
        # 重新解
        s, t = solve_weighted_ls(x, y, w, allow_shift=allow_shift)

    # 数据不足列的回退策略：用全局 s,t 或最近邻填充
    count_per_col = valid_mask.sum(dim=0)  # [W]
    if (count_per_col < min_points).any():
        # 全局稳健拟合（把所有列拼一起）
        w_all = valid_mask.to(dtype)
        s_all, t_all = solve_weighted_ls(
            x.view(-1,1).expand(-1,W).reshape(H,W),
            y.view(-1,1).expand(-1,W).reshape(H,W),
            w_all, allow_shift=allow_shift
        )
        # 用全局值去填补稀疏列
        bad = count_per_col < min_points
        s = torch.where(bad, s_all.mean().expand_as(s), s)
        t = torch.where(bad, t_all.mean().expand_as(t), t)

    # 生成列拟合后的垂直视差估计
    disp_v_fit = s.view(1, W) * x + t.view(1, W)   # [H,W]
    return disp_v_fit.to(dtype), s.to(dtype), t.to(dtype)






def fit_disp_from_depth_inv(monodepth, gt_disp, mask=None, shift=True, eps=1e-6):
    """
    用 x=1/Z 拟合带符号视差：gt_disp ≈ s * (1/monodepth) + (t)
    返回 (s, t) 或 (s, 0)
    """
    # x = 1/Z
    x = 1.0 / (monodepth.clamp_min(eps))
    y = gt_disp

    if mask is None:
        # 允许负视差；排除无效与极小深度/逆深度
        mask = torch.isfinite(x) & torch.isfinite(y) & (x > 0) & (y.abs() > 0)

    x = x[mask].view(-1)
    y = y[mask].view(-1)

    if x.numel() < 50:  # 太少就放弃拟合
        s = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        t = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return s, t

    # 最小二乘
    if shift:
        X = torch.stack([x, torch.ones_like(x)], dim=1)  # [N,2]
        A = X.t() @ X + 1e-6 * torch.eye(2, device=X.device, dtype=X.dtype)
        b = X.t() @ y
        params = torch.linalg.solve(A, b)
        s, t = params[0], params[1]
    else:
        # 仅比例（t=0）
        s = (x @ y) / (x @ x + 1e-6)
        t = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return s, t




class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)   #
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list



class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)




    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list





class Monster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.axis_u = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.args.hidden_dims[i], 1, 1))
            for i in range(self.args.n_gru_layers)
        ])
        self.axis_v = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.args.hidden_dims[i], 1, 1))
            for i in range(self.args.n_gru_layers)
        ])


        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0])


        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_stem_v = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_stem_v.load_state_dict(self.corr_stem.state_dict(), strict=True)

        self.corr_feature_att = FeatureAtt(8, 96)
        self.corr_feature_att_v = FeatureAtt(8, 96)
        self.corr_feature_att_v.load_state_dict(self.corr_feature_att.state_dict(), strict=True)
        self.cost_agg = hourglass(8)
        self.cost_agg_v = hourglass(8)
        self.cost_agg_v.load_state_dict(self.cost_agg.state_dict(), strict=True)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.classifier_v = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.classifier_v.load_state_dict(self.classifier.state_dict(), strict=True)


        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(f'./pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        # state_dict_dpt = torch.load(f'/home/cjd/cvpr2025/fusion/Depth-Anything-V2-list3/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)

        del depth_anything, state_dict_dpt, depth_anything_decoder
        self.REMP = REMP()
        self.REMP_v = REMP_Vertical()  # 新增垂直细化

        # 直接复制 REMP 的权重到 REMP_v（层结构/名字一致，因此 strict=True 也能过）
        self.REMP_v.load_state_dict(self.REMP.state_dict(), strict=True)


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def infer_mono(self, image1, image2):
        height_ori, width_ori = image1.shape[2:]
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono_right = self.mono_decoder(features_right_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return depth_mono, depth_mono_right,[features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, H_invs,iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            depth_mono, depth_mono_right,features_mono_left,  features_mono_right = self.infer_mono(image1, image2)

        scale_factor = 0.25
        size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))

        disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)
        disp_mono_4x_vertical = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)

        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        stem_x_list = [stem_16x, stem_8x, stem_4x]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        # gwc_volume = build_gwc_volume_2d_flat(
        #     match_left, match_right, max_dx=self.args.max_disp//4, max_dy=self.args.max_disp//16, num_groups=8
        # )  # [NEW]

        neg = max(1, (self.args.max_disp // 4) // 16)  # 举例：给少量负视差范围；你也可直接写常数 2/4/8
        pos = self.args.max_disp // 4
        # gwc_volume = build_gwc_volume_signed(match_left, match_right, maxdisp_pos=pos, maxdisp_neg=neg, num_groups=8)
        # gwc_volume_vertical = build_gwc_volume_vertical_signed(match_left, match_right, self.args.max_disp // 16, 8)


        gwc_volume, gwc_volume_vertical = build_depth_enhanced_gwc_volumes(
            match_left, match_right, depth_mono, depth_mono_right,
            maxdisp_h=self.args.max_disp // 4,  # max_disp // 4
            maxdisp_v=self.args.max_disp // 16,  # max_disp // 16
            maxdisp_neg=neg,
            num_groups=8,
            depth_weight=0.3,  # 深度先验权重 λ      depth_weight=0.3,depth_threshold=0.2性能最好  0.555
            depth_threshold=0.2,  # 深度阈值  τ 0.1
            reduction='max'
        )

        D_h = gwc_volume.shape[2]
        D_v = gwc_volume_vertical.shape[2]
        D_max = max(D_h, D_v)

        D_align = int(math.ceil(D_max / 8.0) * 8)
        gwc_volume, D_h_orig = pad_cost_volume_D_right(gwc_volume, D_align)
        gwc_volume_vertical, D_v_orig = pad_cost_volume_D_right(gwc_volume_vertical, D_align)

        gwc_volume = self.corr_stem(gwc_volume) #这个地方共享权重方式进行处理
        gwc_volume_vertical = self.corr_stem_v(gwc_volume_vertical) #这个就是采用3D卷积用来融合的


        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        gwc_volume_vertical = self.corr_feature_att_v(gwc_volume_vertical, features_left[0])


        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
        geo_encoding_volume_vertical = self.cost_agg_v(gwc_volume_vertical, features_left)


        # Init disp from geometry encoding volume
        prob =self.classifier(geo_encoding_volume).squeeze(1)
        prob_vertical = self.classifier_v(geo_encoding_volume_vertical).squeeze(1)

        prob = crop_cost_volume_D_right( prob,D_h_orig)
        prob_vertical = crop_cost_volume_D_right(prob_vertical,D_v_orig)

        tau_h, tau_v = 1.0, 0.7
        prob = F.softmax(prob / tau_h, dim=1)
        prob_vertical = F.softmax(prob_vertical / tau_v, dim=1)

        init_disp = disparity_regression1(prob, maxdisp_pos=pos, maxdisp_neg=neg)  # [B,1,H,W]
        maxdisp_v = self.args.max_disp // 16
        init_disp_vertical = disparity_regression1(prob_vertical, maxdisp_pos=maxdisp_v + 1, maxdisp_neg=maxdisp_v)

        
        del prob, prob_vertical,gwc_volume,gwc_volume_vertical

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        # cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]    #从这个地方进行预测则进行分支预测，让水平和垂直有不同的视差权重
        net_list_vertical = [torch.tanh(x[0]) for x in cnet_list]  #这个地方因为后面单独用得到，所以进行了拆分

        raw_inp = [torch.relu(x[1]) for x in cnet_list]
        raw_inp = [torch.relu(x) for x in raw_inp]


        inp_list = [
            list(conv(inp + self.axis_u[i]).split(split_size=conv.out_channels // 3, dim=1))
            for i, (inp, conv) in enumerate(zip(raw_inp, self.context_zqr_convs))
        ]
        inp_list_vertical = [
            list(conv(inp + self.axis_v[i]).split(split_size=conv.out_channels // 3, dim=1))
            for i, (inp, conv) in enumerate(zip(raw_inp, self.context_zqr_convs))
        ]

        net_list_mono = [x.clone() for x in net_list]
        net_list_mono_vertical = [x.clone() for x in net_list_vertical]

        geo_block = Combined_Geo_Encoding_Volume
        geo_block_vertical = Combined_Geo_Encoding_Volume_Vertical


        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        geo_fn_vertical = geo_block_vertical(match_left.float(), match_right.float(),
                                             geo_encoding_volume_vertical.float(), radius=self.args.corr_radius,
                                             num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape

        coords_x = torch.arange(w, device=match_left.device, dtype=torch.float32).view(1, 1, w, 1).repeat(b, h, 1, 1)
        coords_y = torch.arange(h, device=match_left.device, dtype=torch.float32).view(1, h, 1, 1).repeat(b, 1, w, 1)


        disp = init_disp
        disp_vertical = init_disp_vertical
        disp_preds = []
        disp_preds_vertical = []
        disp_perds_gt = []
        for itr in range(iters):
            disp = disp.detach()
            disp_vertical = disp_vertical.detach()
            if itr >= int(1):
                disp_mono_4x = disp_mono_4x.detach()
                disp_mono_4x_vertical= disp_mono_4x_vertical.detach()
            geo_feat = geo_fn(disp, coords_x)
            geo_feat_vertical = geo_fn_vertical(disp_vertical, coords_y)
            #geo_feat, geo_feat_vertical = geo_fn_combined(disp, disp_vertical, coords_combined)

            if itr > int(iters-8):
                if itr == int(iters-7):
                    bs, _, _, _ = disp.shape
                    for i in range(bs):
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            aligned_h, aligned_v = align_with_enhanced_method(
                                disp_mono_4x[i], disp[i], disp_vertical[i],
                                patch_size=16,
                                overlap=0.5,
                                verbose=False
                            )
                            disp_mono_4x[i]=aligned_h
                            disp_mono_4x_vertical[i]=aligned_v


                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]
                warped_right_mono_vertical = disp_warp_vertical(features_right[0], disp_mono_4x_vertical.clone().to(features_right[0].dtype))[0]
                flaw_mono = warped_right_mono - features_left[0]
                flaw_mono_vertical = warped_right_mono_vertical - features_left[0]

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]
                warped_right_stereo_vertical = disp_warp_vertical(features_right[0], disp_vertical.clone().to(features_right[0].dtype))[0]
                flaw_stereo = warped_right_stereo - features_left[0]
                flaw_stereo_vertical = warped_right_stereo_vertical - features_left[0]    #垂直视差的话这个地方有大问题，不能这样调用，这个地方也应该采用组合2D的方式
                geo_feat_mono = geo_fn(disp_mono_4x, coords_x)
                geo_feat_mono_vertical = geo_fn_vertical(disp_mono_4x_vertical, coords_y)
                #geo_feat_mono, geo_feat_mono_vertical = geo_fn_combined(disp_mono_4x, disp_mono_4x_vertical, coords_combined)

            if itr <= int(iters-8):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                net_list_vertical, mask_feat_4_vertical, delta_disp_vertical = self.update_block(net_list_vertical, inp_list_vertical, geo_feat_vertical, disp_vertical,
                                                                      iter16=self.args.n_gru_layers == 3,
                                                                      iter08=self.args.n_gru_layers >= 2)
            else:
                net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_4x, geo_feat_mono, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                net_list_vertical, mask_feat_4_vertical, delta_disp_vertical = self.update_block_mix_stereo(net_list_vertical, inp_list_vertical, flaw_stereo_vertical, disp_vertical,
                                                                                 geo_feat_vertical, flaw_mono_vertical, disp_mono_4x_vertical,
                                                                                 geo_feat_mono_vertical,
                                                                                 iter16=self.args.n_gru_layers == 3,
                                                                                 iter08=self.args.n_gru_layers >= 2)
                net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list,
                                                                                              flaw_mono, disp_mono_4x,
                                                                                              geo_feat_mono,
                                                                                              flaw_stereo, disp,
                                                                                              geo_feat,
                                                                                              iter16=self.args.n_gru_layers == 3,
                                                                                              iter08=self.args.n_gru_layers >= 2)
                net_list_mono_vertical, mask_feat_4_mono_vertical, delta_disp_mono_vertical = self.update_block_mix_mono(net_list_mono_vertical, inp_list_vertical,
                                                                                              flaw_mono_vertical, disp_mono_4x_vertical,
                                                                                              geo_feat_mono_vertical,
                                                                                              flaw_stereo_vertical, disp_vertical,
                                                                                              geo_feat_vertical,
                                                                                              iter16=self.args.n_gru_layers == 3, iter08=self.args.n_gru_layers >= 2)

                disp_mono_4x = disp_mono_4x + delta_disp_mono
                disp_mono_4x_vertical = disp_mono_4x_vertical + delta_disp_mono_vertical
                disp_mono_4x_up = self.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
                disp_mono_4x_up_vertical = self.upsample_disp(disp_mono_4x_vertical, mask_feat_4_mono_vertical, stem_2x)

                recover1= inverse_transform_verification1_vectorized(disp_mono_4x_up,disp_mono_4x_up_vertical,H_invs)
                disp_preds.append(disp_mono_4x_up)
                disp_preds_vertical.append(disp_mono_4x_up_vertical)
                disp_perds_gt.append(recover1)

            disp = disp + delta_disp
            disp_vertical = disp_vertical + delta_disp_vertical
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_up_vertical = self.upsample_disp(disp_vertical, mask_feat_4_vertical, stem_2x)
            recover2 = inverse_transform_verification1_vectorized(disp_up,disp_up_vertical,H_invs)

            if itr == iters - 1:
                refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
                disp_up = disp_up + refine_value
                refine_value_v = self.REMP_v(disp_mono_4x_up_vertical, disp_up_vertical, image1, image2)
                disp_up_vertical = disp_up_vertical + refine_value_v
                recover2 = inverse_transform_verification1_vectorized(disp_up,disp_up_vertical,H_invs)
            disp_preds.append(disp_up)
            disp_preds_vertical.append(disp_up_vertical)
            disp_perds_gt.append(recover2)

        if test_mode:
            return disp_up,disp_up_vertical,recover2
        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)
        init_disp_vertical = context_upsample(init_disp_vertical * 4., spx_pred.float()).unsqueeze(1)
        recover_dirsty_init = inverse_transform_verification1_vectorized(init_disp, init_disp_vertical, H_invs)
        return init_disp,init_disp_vertical,recover_dirsty_init,disp_preds,disp_preds_vertical,disp_perds_gt,depth_mono