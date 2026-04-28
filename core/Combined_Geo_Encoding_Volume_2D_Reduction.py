import torch
import torch.nn.functional as F


def corr_2d_depth_enhanced(fmap1, fmap2, depth_left, depth_right, max_disp_h, max_disp_v,
                           depth_weight=0.3, depth_threshold=0.1):
    """
    深度增强的2D correlation计算
    Args:
        fmap1: 左相机特征图 [B, D, H, W]
        fmap2: 右相机特征图 [B, D, H, W]
        depth_left: 左相机深度图 [B, 1, H_d, W_d]
        depth_right: 右相机深度图 [B, 1, H_d, W_d]
        max_disp_h: 水平最大视差
        max_disp_v: 垂直最大视差
        depth_weight: 深度先验权重
        depth_threshold: 深度阈值
    Returns:
        corr_volume: 深度增强的相关性体 [B, H, W, max_disp_v, max_disp_h]
    """
    B, D, H, W = fmap1.shape

    # 将深度图调整到与特征图相同的尺寸
    depth_left_resized = F.interpolate(depth_left, size=(H, W), mode='bilinear', align_corners=False)
    depth_right_resized = F.interpolate(depth_right, size=(H, W), mode='bilinear', align_corners=False)

    corr_volume = fmap1.new_zeros([B, H, W, max_disp_v, max_disp_h])

    for dy in range(max_disp_v):
        for dx in range(max_disp_h):
            if dy == 0 and dx == 0:
                # 计算特征相关性
                feature_corr = torch.einsum('bdhw,bdhw->bhw', fmap1, fmap2)

                # 计算深度一致性
                depth_diff = torch.abs(depth_left_resized.squeeze(1) - depth_right_resized.squeeze(1))
                depth_similarity = torch.exp(-depth_diff / depth_threshold)

                # 融合特征相关性和深度先验
                enhanced_corr = feature_corr * (1 + depth_weight * depth_similarity)
                corr_volume[:, :, :, dy, dx] = enhanced_corr

            elif dy == 0:
                valid_w = W - dx
                # 特征相关性
                feature_corr = torch.einsum('bdhw,bdhw->bhw',
                                            fmap1[:, :, :, :valid_w],
                                            fmap2[:, :, :, dx:])

                # 深度一致性
                depth_diff = torch.abs(
                    depth_left_resized[:, :, :, :valid_w].squeeze(1) -
                    depth_right_resized[:, :, :, dx:].squeeze(1)
                )
                depth_similarity = torch.exp(-depth_diff / depth_threshold)

                # 融合
                enhanced_corr = feature_corr * (1 + depth_weight * depth_similarity)
                corr_volume[:, :, :valid_w, dy, dx] = enhanced_corr

            elif dx == 0:
                valid_h = H - dy
                # 特征相关性
                feature_corr = torch.einsum('bdhw,bdhw->bhw',
                                            fmap1[:, :, :valid_h, :],
                                            fmap2[:, :, dy:, :])

                # 深度一致性
                depth_diff = torch.abs(
                    depth_left_resized[:, :, :valid_h, :].squeeze(1) -
                    depth_right_resized[:, :, dy:, :].squeeze(1)
                )
                depth_similarity = torch.exp(-depth_diff / depth_threshold)

                # 融合
                enhanced_corr = feature_corr * (1 + depth_weight * depth_similarity)
                corr_volume[:, :valid_h, :, dy, dx] = enhanced_corr

            else:
                valid_h = H - dy
                valid_w = W - dx
                # 特征相关性
                feature_corr = torch.einsum('bdhw,bdhw->bhw',
                                            fmap1[:, :, :valid_h, :valid_w],
                                            fmap2[:, :, dy:, dx:])

                # 深度一致性
                depth_diff = torch.abs(
                    depth_left_resized[:, :, :valid_h, :valid_w].squeeze(1) -
                    depth_right_resized[:, :, dy:, dx:].squeeze(1)
                )
                depth_similarity = torch.exp(-depth_diff / depth_threshold)

                # 融合
                enhanced_corr = feature_corr * (1 + depth_weight * depth_similarity)
                corr_volume[:, :valid_h, :valid_w, dy, dx] = enhanced_corr

    return corr_volume.contiguous()


# 使用示例：
# 在原始类中，只需要将corr_2d替换为corr_2d_depth_enhanced

class Combined_Geo_Encoding_Volume_2D_Reduction_DepthEnhanced:
    def __init__(self, init_fmap1, init_fmap2, geo_volume_h, geo_volume_v,
                 depth_left, depth_right, max_disp_h, max_disp_v,  # 新增depth参数
                 num_levels=2, radius=4, reduction='max',
                 depth_weight=0.3, depth_threshold=0.1):  # 新增深度参数
        """
        深度增强的几何编码体 - 最小修改版本
        只在corr_2d函数中加入深度先验，其他保持不变
        """
        self.num_levels = num_levels
        self.radius = radius
        self.reduction = reduction
        self.depth_weight = depth_weight  # 新增
        self.depth_threshold = depth_threshold  # 新增

        self.geo_volume_h_pyramid = []
        self.geo_volume_v_pyramid = []
        self.init_corr_pyramid = []

        # 使用深度增强的2D correlation（唯一的主要修改）
        init_corr = corr_2d_depth_enhanced(
            init_fmap1, init_fmap2, depth_left, depth_right,  # 新增depth参数
            max_disp_h=max_disp_h, max_disp_v=max_disp_v,
            depth_weight=depth_weight, depth_threshold=depth_threshold  # 新增参数
        )

        b, h, w, max_disp_v, max_disp_h = init_corr.shape

        # 以下代码完全保持不变
        # 处理几何编码体
        b_h, c_h, d_h, h_h, w_h = geo_volume_h.shape
        geo_volume_h = geo_volume_h.permute(0, 3, 4, 1, 2).reshape(b * h * w, c_h, 1, d_h)

        b_v, c_v, d_v, h_v, w_v = geo_volume_v.shape
        geo_volume_v = geo_volume_v.permute(0, 3, 4, 1, 2).reshape(b * h * w, c_v, 1, d_v)

        # 重塑init_corr为适合2D搜索的形式
        init_corr = init_corr.reshape(b * h * w, 1, max_disp_v, max_disp_h)

        self.geo_volume_h_pyramid.append(geo_volume_h)
        self.geo_volume_v_pyramid.append(geo_volume_v)
        self.init_corr_pyramid.append(init_corr)

        # 构建金字塔
        for i in range(self.num_levels - 1):
            geo_volume_h = F.avg_pool2d(geo_volume_h, [1, 2], stride=[1, 2])
            geo_volume_v = F.avg_pool2d(geo_volume_v, [1, 2], stride=[1, 2])
            self.geo_volume_h_pyramid.append(geo_volume_h)
            self.geo_volume_v_pyramid.append(geo_volume_v)

        for i in range(self.num_levels - 1):
            init_corr = F.avg_pool2d(init_corr, [2, 2], stride=[2, 2])
            self.init_corr_pyramid.append(init_corr)

        # 如果使用learned reduction，初始化权重
        if reduction == 'learned':
            self.h_weights = torch.nn.Parameter(torch.ones(1, 1, 2 * radius + 1, 1) / (2 * radius + 1))
            self.v_weights = torch.nn.Parameter(torch.ones(1, 1, 1, 2 * radius + 1) / (2 * radius + 1))

    # 以下所有方法保持完全不变
    def __call__(self, disp_h, disp_v, coords):
        """
        执行2D搜索，然后通过降维提取水平和垂直代价
        （此方法保持完全不变）
        """
        r = self.radius
        b, _, h, w = disp_h.shape

        device = disp_h.device
        coords = coords.to(device)
        disp_v = disp_v.to(device)

        coords_x = coords[:, 0:1]
        coords_y = coords[:, 1:2]

        out_pyramid_h = []
        out_pyramid_v = []

        for i in range(self.num_levels):
            geo_volume_h = self.geo_volume_h_pyramid[i]
            geo_volume_v = self.geo_volume_v_pyramid[i]

            # 创建2D搜索网格
            device = disp_h.device
            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)

            # 创建2D网格 [2*r+1, 2*r+1, 2]
            grid_y, grid_x = torch.meshgrid(dy, dx, indexing='ij')
            search_grid = torch.stack([grid_x, grid_y], dim=-1)  # [2*r+1, 2*r+1, 2]

            # 保存原始形状用于后续reshape
            grid_shape = (2 * r + 1, 2 * r + 1)

            # 重塑为 [1, (2*r+1)^2, 1, 2]
            search_grid_flat = search_grid.reshape(1, -1, 1, 2)

            # 添加当前视差估计
            disp_current = torch.stack([
                disp_h.reshape(b * h * w, 1, 1, 1) / 2 ** i,
                disp_v.reshape(b * h * w, 1, 1, 1) / 2 ** i
            ], dim=-1).squeeze(-2)

            # 2D搜索位置
            search_positions = search_grid_flat + disp_current

            # ===== 几何编码体的2D采样 =====
            # 从水平几何编码体采样
            geo_h_positions = torch.cat([search_positions[..., 0:1],
                                         torch.zeros_like(search_positions[..., 0:1])], dim=-1)
            geo_h_sampled = self.bilinear_sampler(geo_volume_h, geo_h_positions)
            geo_h_sampled = geo_h_sampled.view(b * h * w, -1, grid_shape[0], grid_shape[1])

            # 从垂直几何编码体采样
            geo_v_positions = torch.cat([torch.zeros_like(search_positions[..., 1:2]),
                                         search_positions[..., 1:2]], dim=-1)
            geo_v_sampled = self.bilinear_sampler(geo_volume_v, geo_v_positions)
            geo_v_sampled = geo_v_sampled.view(b * h * w, -1, grid_shape[0], grid_shape[1])

            # ===== init_corr的2D采样 =====
            init_corr = self.init_corr_pyramid[i]

            coords_current = torch.stack([
                coords_x.reshape(b * h * w, 1, 1, 1) / 2 ** i,
                coords_y.reshape(b * h * w, 1, 1, 1) / 2 ** i
            ], dim=-1).squeeze(-2)

            init_search_positions = coords_current - disp_current + search_grid_flat

            init_corr_sampled = self.bilinear_sampler(init_corr, init_search_positions)
            init_corr_sampled = init_corr_sampled.view(b * h * w, -1, grid_shape[0], grid_shape[1])

            # ===== 应用降维策略 =====
            # 对于水平代价：沿垂直方向(dim=2)降维
            geo_h_reduced = self._reduce_volume(geo_h_sampled, dim=2, direction='horizontal')
            init_corr_h_reduced = self._reduce_volume(init_corr_sampled, dim=2, direction='horizontal')

            # 对于垂直代价：沿水平方向(dim=3)降维
            geo_v_reduced = self._reduce_volume(geo_v_sampled, dim=3, direction='vertical')
            init_corr_v_reduced = self._reduce_volume(init_corr_sampled, dim=3, direction='vertical')

            # reshape回原始batch形状
            geo_h_reduced = geo_h_reduced.reshape(b, h, w, -1)
            init_corr_h_reduced = init_corr_h_reduced.reshape(b, h, w, -1)
            geo_v_reduced = geo_v_reduced.reshape(b, h, w, -1)
            init_corr_v_reduced = init_corr_v_reduced.reshape(b, h, w, -1)

            out_pyramid_h.append(geo_h_reduced)
            out_pyramid_h.append(init_corr_h_reduced)
            out_pyramid_v.append(geo_v_reduced)
            out_pyramid_v.append(init_corr_v_reduced)

        out_h = torch.cat(out_pyramid_h, dim=-1)
        out_v = torch.cat(out_pyramid_v, dim=-1)

        out_h = out_h.permute(0, 3, 1, 2).contiguous().float()
        out_v = out_v.permute(0, 3, 1, 2).contiguous().float()

        return out_h, out_v

    def _reduce_volume(self, volume, dim, direction):
        """
        对2D搜索结果进行降维（保持不变）
        """
        if self.reduction == 'max':
            reduced, _ = torch.max(volume, dim=dim)
        elif self.reduction == 'mean':
            reduced = torch.mean(volume, dim=dim)
        elif self.reduction == 'weighted':
            # 使用距离加权平均
            r = self.radius
            device = volume.device
            if direction == 'horizontal':
                weights = torch.exp(-torch.abs(torch.linspace(-r, r, 2 * r + 1, device=device)) / r)
                weights = weights / weights.sum()
                weights = weights.view(1, 1, -1, 1)
                reduced = (volume * weights).sum(dim=dim)
            else:
                weights = torch.exp(-torch.abs(torch.linspace(-r, r, 2 * r + 1, device=device)) / r)
                weights = weights / weights.sum()
                weights = weights.view(1, 1, 1, -1)
                reduced = (volume * weights).sum(dim=dim)
        elif self.reduction == 'learned':
            if direction == 'horizontal':
                if self.h_weights is None:
                    self.h_weights = torch.ones(1, 1, 2 * self.radius + 1, 1, device=volume.device) / (
                            2 * self.radius + 1)
                else:
                    self.h_weights = self.h_weights.to(volume.device)
                reduced = (volume * self.h_weights).sum(dim=dim)
            else:
                if self.v_weights is None:
                    self.v_weights = torch.ones(1, 1, 1, 2 * self.radius + 1, device=volume.device) / (
                            2 * self.radius + 1)
                else:
                    self.v_weights = self.v_weights.to(volume.device)
                reduced = (volume * self.v_weights).sum(dim=dim)
        elif self.reduction == 'center':
            # 只取中心线
            center = self.radius
            if dim == 2:  # horizontal: 取中心行
                reduced = volume[:, :, center, :].contiguous()
            else:  # vertical: 取中心列
                reduced = volume[:, :, :, center].contiguous()
        elif self.reduction == 'adaptive':
            # 自适应降维：基于相关性强度选择
            corr_strength = torch.abs(volume)
            if dim == 2:  # horizontal
                weights = F.softmax(corr_strength.mean(dim=1, keepdim=True), dim=2)
                reduced = (volume * weights).sum(dim=2)
            else:  # vertical
                weights = F.softmax(corr_strength.mean(dim=1, keepdim=True), dim=3)
                reduced = (volume * weights).sum(dim=3)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

        return reduced

    @staticmethod
    def bilinear_sampler(img, coords):
        """双线性采样（保持不变）"""
        return F.grid_sample(img, coords, mode='bilinear',
                             padding_mode='zeros', align_corners=True)