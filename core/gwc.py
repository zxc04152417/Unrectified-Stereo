import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def groupwise_correlation(fea1, fea2, num_groups):
    """
    原始的分组相关性计算（保持不变）
    """
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def depth_guided_correlation(ref_fea, target_fea, ref_depth, target_depth, num_groups,
                             depth_weight=0.3, depth_threshold=0.1):
    """
    融合深度先验的相关性计算
    Args:
        ref_fea: 参考特征图 [B, C, H, W]
        target_fea: 目标特征图 [B, C, H, W]
        ref_depth: 参考深度图 [B, 1, H, W]
        target_depth: 目标深度图 [B, 1, H, W]
        num_groups: 分组数
        depth_weight: 深度先验权重
        depth_threshold: 深度差异阈值
    Returns:
        correlation: 融合深度先验的相关性 [B, num_groups, H, W]
    """
    # 原始特征相关性
    feature_corr = groupwise_correlation(ref_fea, target_fea, num_groups)

    # 深度一致性约束
    depth_diff = torch.abs(ref_depth - target_depth)  # [B, 1, H, W]

    # 深度相似性权重：深度差越小，权重越大
    depth_similarity = torch.exp(-depth_diff / depth_threshold)  # [B, 1, H, W]

    # 将深度权重扩展到所有组
    depth_weight_expanded = depth_similarity.expand(-1, num_groups, -1, -1)  # [B, num_groups, H, W]

    # 融合特征相关性和深度先验
    # 方式1: 加权融合
    enhanced_corr = feature_corr * (1 + depth_weight * depth_weight_expanded)

    # 方式2: 深度一致性惩罚（可选）
    # depth_penalty = -depth_weight * depth_diff.expand(-1, num_groups, -1, -1)
    # enhanced_corr = feature_corr + depth_penalty

    return enhanced_corr


def build_2d_gwc_volume_with_depth_prior(ref_fea, target_fea, ref_depth, target_depth,
                                         maxdisp_h, maxdisp_v, num_groups,
                                         depth_weight=0.3, depth_threshold=0.1):
    """
    构建融合深度先验的二维GWC代价体
    Args:
        ref_fea: 参考特征图 [B, C, H, W]
        target_fea: 目标特征图 [B, C, H, W]
        ref_depth: 参考深度图 [B, 1, H, W]
        target_depth: 目标深度图 [B, 1, H, W]
        maxdisp_h: 水平方向最大视差
        maxdisp_v: 垂直方向最大视差
        num_groups: 分组数
        depth_weight: 深度先验权重
        depth_threshold: 深度差异阈值
    Returns:
        volume_2d: 二维代价体 [B, num_groups, maxdisp_v, maxdisp_h, H, W]
    """
    B, C, H, W = ref_fea.shape
    volume_2d = ref_fea.new_zeros([B, num_groups, maxdisp_v, maxdisp_h, H, W])

    for dy in range(maxdisp_v):
        for dx in range(maxdisp_h):
            if dy == 0 and dx == 0:
                # 无位移
                volume_2d[:, :, dy, dx, :, :] = depth_guided_correlation(
                    ref_fea, target_fea, ref_depth, target_depth,
                    num_groups, depth_weight, depth_threshold
                )
            elif dy == 0:
                # 仅水平位移
                if dx < W:
                    volume_2d[:, :, dy, dx, :, dx:] = depth_guided_correlation(
                        ref_fea[:, :, :, dx:],
                        target_fea[:, :, :, :-dx],
                        ref_depth[:, :, :, dx:],
                        target_depth[:, :, :, :-dx],
                        num_groups, depth_weight, depth_threshold
                    )
            elif dx == 0:
                # 仅垂直位移
                if dy < H:
                    volume_2d[:, :, dy, dx, dy:, :] = depth_guided_correlation(
                        ref_fea[:, :, dy:, :],
                        target_fea[:, :, :-dy, :],
                        ref_depth[:, :, dy:, :],
                        target_depth[:, :, :-dy, :],
                        num_groups, depth_weight, depth_threshold
                    )
            else:
                # 水平和垂直同时位移
                if dy < H and dx < W:
                    volume_2d[:, :, dy, dx, dy:, dx:] = depth_guided_correlation(
                        ref_fea[:, :, dy:, dx:],
                        target_fea[:, :, :-dy, :-dx],
                        ref_depth[:, :, dy:, dx:],
                        target_depth[:, :, :-dy, :-dx],
                        num_groups, depth_weight, depth_threshold
                    )

    return volume_2d.contiguous()


#下面这个是带有副的水平视差的。
import torch

def build_2d_gwc_volume_with_depth_prior_signed(
        ref_fea, target_fea, ref_depth, target_depth,
        maxdisp_h_pos, maxdisp_h_neg, maxdisp_v, num_groups,
        depth_weight=0.3, depth_threshold=0.1):
    """
    构建融合深度先验的二维 GWC 代价体（水平/垂直都支持 signed search）

    视差定义：
      d_h = x_L - x_R
      d_v = y_L - y_R

    因此：
      x_R = x_L - d_h
      y_R = y_L - d_v

    搜索范围：
      dx ∈ [-maxdisp_h_neg, ..., -1, 0, 1, ..., maxdisp_h_pos-1]
      dy ∈ [-maxdisp_v, ..., 0, ..., +maxdisp_v]

    Args:
        ref_fea:       左/参考特征图 [B, C, H, W]
        target_fea:    右/目标特征图 [B, C, H, W]
        ref_depth:     左/参考深度图 [B, 1, H, W]
        target_depth:  右/目标深度图 [B, 1, H, W]
        maxdisp_h_pos: 水平正向视差平面数（包含 0 平面时，正向实际搜索为 1..maxdisp_h_pos-1）
        maxdisp_h_neg: 水平负向最大绝对视差（搜索 -maxdisp_h_neg..-1）
        maxdisp_v:     垂直最大绝对视差（搜索 -maxdisp_v..+maxdisp_v）
        num_groups:    GWC 分组数
        depth_weight:  深度先验权重
        depth_threshold: 深度差阈值（具体作用依赖 depth_guided_correlation 的实现）

    Returns:
        volume_2d:
            [B, num_groups, Dv, Dh, H, W]
            其中
              Dv = 2*maxdisp_v + 1
              Dh = maxdisp_h_neg + maxdisp_h_pos
    """
    B, C, H, W = ref_fea.shape

    Dv = 2 * maxdisp_v + 1
    Dh = maxdisp_h_neg + maxdisp_h_pos

    volume_2d = ref_fea.new_zeros([B, num_groups, Dv, Dh, H, W])

    def v_plane_idx(dy: int) -> int:
        # dy ∈ [-maxdisp_v, ..., +maxdisp_v] -> [0, ..., Dv-1]
        return dy + maxdisp_v

    def h_plane_idx(dx: int) -> int:
        # dx ∈ [-maxdisp_h_neg, ..., maxdisp_h_pos-1] -> [0, ..., Dh-1]
        return dx + maxdisp_h_neg

    for dy in range(-maxdisp_v, maxdisp_v + 1):
        He = H - abs(dy)
        if He <= 0:
            continue

        # 垂直方向有效起点
        ys_ref = max(dy, 0)
        ys_tgt = max(-dy, 0)

        for dx in range(-maxdisp_h_neg, maxdisp_h_pos):
            We = W - abs(dx)
            if We <= 0:
                continue

            # 水平方向有效起点
            ys_ref = max(dy, 0)
            ys_tgt = max(-dy, 0)
            xs_ref = max(dx, 0)
            xs_tgt = max(-dx, 0)

            ref_patch = ref_fea[:, :, ys_ref:ys_ref + He, xs_ref:xs_ref + We]
            tgt_patch = target_fea[:, :, ys_tgt:ys_tgt + He, xs_tgt:xs_tgt + We]

            ref_depth_patch = ref_depth[:, :, ys_ref:ys_ref + He, xs_ref:xs_ref + We]
            tgt_depth_patch = target_depth[:, :, ys_tgt:ys_tgt + He, xs_tgt:xs_tgt + We]

            cost = depth_guided_correlation(
                ref_patch.contiguous(),
                tgt_patch.contiguous(),
                ref_depth_patch.contiguous(),
                tgt_depth_patch.contiguous(),
                num_groups=num_groups,
                depth_weight=depth_weight,
                depth_threshold=depth_threshold
            )  # [B, num_groups, He, We]

            volume_2d[:, :, v_plane_idx(dy), h_plane_idx(dx),
                      ys_ref:ys_ref + He, xs_ref:xs_ref + We] = cost

    return volume_2d.contiguous()











def depth_consistency_volume(ref_depth, target_depth, maxdisp_h, maxdisp_v):
    """
    构建纯深度一致性代价体（可选的额外信息）
    Args:
        ref_depth: 参考深度图 [B, 1, H, W]
        target_depth: 目标深度图 [B, 1, H, W]
        maxdisp_h: 水平最大视差
        maxdisp_v: 垂直最大视差
    Returns:
        depth_volume: 深度一致性代价体 [B, 1, maxdisp_v, maxdisp_h, H, W]
    """
    B, C, H, W = ref_depth.shape
    depth_volume = ref_depth.new_zeros([B, 1, maxdisp_v, maxdisp_h, H, W])

    for dy in range(maxdisp_v):
        for dx in range(maxdisp_h):
            if dy == 0 and dx == 0:
                depth_volume[:, :, dy, dx, :, :] = -torch.abs(ref_depth - target_depth)
            elif dy == 0:
                if dx < W:
                    depth_volume[:, :, dy, dx, :, dx:] = -torch.abs(
                        ref_depth[:, :, :, dx:] - target_depth[:, :, :, :-dx]
                    )
            elif dx == 0:
                if dy < H:
                    depth_volume[:, :, dy, dx, dy:, :] = -torch.abs(
                        ref_depth[:, :, dy:, :] - target_depth[:, :, :-dy, :]
                    )
            else:
                if dy < H and dx < W:
                    depth_volume[:, :, dy, dx, dy:, dx:] = -torch.abs(
                        ref_depth[:, :, dy:, dx:] - target_depth[:, :, :-dy, :-dx]
                    )

    return depth_volume.contiguous()


def extract_horizontal_vertical_volumes(volume_2d, reduction='max'):
    """
    从二维代价体中提取水平和垂直代价体
    Args:
        volume_2d: 二维代价体 [B, num_groups, maxdisp_v, maxdisp_h, H, W]
        reduction: 降维方式，'max' 或 'mean'
    Returns:
        volume_h: 水平代价体 [B, num_groups, maxdisp_h, H, W]
        volume_v: 垂直代价体 [B, num_groups, maxdisp_v, H, W]
    """
    B, num_groups, maxdisp_v, maxdisp_h, H, W = volume_2d.shape

    if reduction == 'max':
        volume_h, _ = torch.max(volume_2d, dim=2)
        volume_v, _ = torch.max(volume_2d, dim=3)
    elif reduction == 'mean':
        volume_h = torch.mean(volume_2d, dim=2)
        volume_v = torch.mean(volume_2d, dim=3)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")

    return volume_h, volume_v


def build_depth_enhanced_gwc_volumes(match_left, match_right, depth_left, depth_right,
                                     maxdisp_h, maxdisp_v, maxdisp_neg,num_groups,
                                     depth_weight=0.3, depth_threshold=0.1,
                                     reduction='max', use_depth_volume=False):
    """
    构建融合深度先验的GWC代价体
    Args:
        match_left: 左相机特征图 [B, C, H, W]
        match_right: 右相机特征图 [B, C, H, W]
        depth_left: 左相机深度图 [B, 1, H_d, W_d]
        depth_right: 右相机深度图 [B, 1, H_d, W_d]
        maxdisp_h: 水平最大视差
        maxdisp_v: 垂直最大视差
        num_groups: 分组数
        depth_weight: 深度先验权重
        depth_threshold: 深度差异阈值
        reduction: 降维方式
        use_depth_volume: 是否额外返回深度一致性代价体
    Returns:
        volume_h: 水平代价体 [B, num_groups, maxdisp_h, H, W]
        volume_v: 垂直代价体 [B, num_groups, maxdisp_v, H, W]
        depth_volume_h: 深度水平代价体（可选）
        depth_volume_v: 深度垂直代价体（可选）
    """
    # 将深度图调整到与特征图相同的尺寸
    _, _, H_fea, W_fea = match_left.shape
    depth_left_resized = F.interpolate(depth_left, size=(H_fea, W_fea),
                                       mode='bilinear', align_corners=False)
    depth_right_resized = F.interpolate(depth_right, size=(H_fea, W_fea),
                                        mode='bilinear', align_corners=False)

    # 构建融合深度先验的二维代价体
    volume_2d = build_2d_gwc_volume_with_depth_prior_signed(
        match_left, match_right, depth_left_resized, depth_right_resized,
        maxdisp_h, maxdisp_neg, maxdisp_v, num_groups, depth_weight, depth_threshold
    )

    # 提取水平和垂直代价体
    volume_h, volume_v = extract_horizontal_vertical_volumes(volume_2d, reduction)

    if use_depth_volume:
        # 额外构建纯深度一致性代价体
        depth_volume_2d = depth_consistency_volume(
            depth_left_resized, depth_right_resized, maxdisp_h, maxdisp_v
        )
        depth_volume_h, depth_volume_v = extract_horizontal_vertical_volumes(
            depth_volume_2d, reduction
        )
        return volume_h, volume_v, depth_volume_h, depth_volume_v

    return volume_h, volume_v