import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)


    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr


class Combined_Geo_Encoding_Volume_Vertical:
    """
    垂直视差用的几何编码：
    - 初始相关 corr 沿高度 H 做 all-pairs：同一列 w 上的 (h, h')
    - geo_volume 仍 reshape 成 [B*H*W, C, 1, D]，在 x 维上对 D 做 1D 采样（与水平对称）
    - 传入 coords 是 y 基坐标 (coords_y)，与 disp_vertical 配合
    """
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # 1) 垂直 all-pairs 相关：固定列 w，在高度维做匹配
        init_corr = Combined_Geo_Encoding_Volume_Vertical.corr(init_fmap1, init_fmap2)
        # init_corr: [B, H1, W, 1, H2]

        # 2) 把 [B, C, D, H, W] 的 geo_volume 摊平成 [B*H*W, C, 1, D]，
        #    这样后面可以在“宽度维= D”上用 bilinear_sampler 1D 采样（与水平保持一致的实现）
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1, d)

        # 3) init_corr 也摊平为 [B*H*W, 1, 1, H2]，方便在“宽度维= H2”上 1D 采样
        _, _, _, _, h2 = init_corr.shape
        init_corr = init_corr.reshape(b * h * w, 1, 1, h2)

        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)

        # 4) 金字塔：仅沿“宽度维”（这里分别是 D 和 H2）做 1/2 下采样
        for _ in range(self.num_levels - 1):
            geo_volume = F.avg_pool2d(geo_volume, kernel_size=[1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels - 1):
            init_corr = F.avg_pool2d(init_corr, kernel_size=[1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp_v, coords_y):
        """
        disp_v:   [B, 1, H, W]   垂直视差
        coords_y: [B, H, W, 1]   每个像素的“行号”基坐标
        返回: [B, C_out, H, W]   （与水平实现保持一致的通道拼接）
        """
        r = self.radius
        b, _, h, w = disp_v.shape
        out_pyramid = []

        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]   # [BHW, C, 1, D_i]
            init_corr  = self.init_corr_pyramid[i]    # [BHW, 1, 1, H2_i]

            # —— 在 geo_volume 上：沿“D_i”维用 1D 采样，窗口为 [-r..r]
            dy = torch.linspace(-r, r, 2 * r + 1, device=disp_v.device, dtype=disp_v.dtype)
            dy = dy.view(1, 1, 2 * r + 1, 1)  # [1,1,W_out=2r+1,1]

            # 注意：输入是 [*, 1, D_i]，因此 grid 的 y=0，x 才是沿 D_i 的“横坐标”
            x0 = dy + disp_v.reshape(b * h * w, 1, 1, 1) / (2 ** i)   # 把 v 视差映射到 D 轴采样
            y0 = torch.zeros_like(x0)
            grid_geo = torch.cat([x0, y0], dim=-1)                   # [BHW, 1, 2r+1, 2]
            geo_smpl = bilinear_sampler(geo_volume, grid_geo).view(b, h, w, -1)

            # —— 在 init_corr 上：沿“对端高度 H2_i”维做 1D 采样（用 y 基坐标 - v + dy）
            init_x0 = (coords_y.reshape(b * h * w, 1, 1, 1) / (2 ** i)   # 基坐标（行号）
                       - disp_v.reshape(b * h * w, 1, 1, 1) / (2 ** i)    # 垂直视差
                       + dy)                                              # 邻域偏移
            init_grid = torch.cat([init_x0, y0], dim=-1)                  # 同样是 x 轴在扫
            init_smpl = bilinear_sampler(init_corr, init_grid).view(b, h, w, -1)

            out_pyramid.append(geo_smpl)
            out_pyramid.append(init_smpl)

        out = torch.cat(out_pyramid, dim=-1)          # [B, H, W, Csum]
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """
        垂直 all-pairs 相关：固定 w，在高度维做匹配。
        输入: fmap1/2: [B, C, H, W]
        输出: [B, H1, W, 1, H2]  （与水平的 [B, H, W1, 1, W2] 对称）
        """
        B, C, H, W = fmap1.shape
        # 对每个列 w：f1[:, h, w] 与 f2[:, h', w] 点积
        corr = torch.einsum('bchw,bcyw->bhwy', fmap1, fmap2)  # [B, H, W, H2]
        corr = corr.reshape(B, H, W, 1, H)                    # H2 == H（若两幅图等高；若不等高就用 fmap2 的 H）
        return corr.contiguous()





#
# class Combined_Geo_Encoding_Volume_Vertical:
#     def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
#         # 初始化部分完全相同，不需要修改
#         self.num_levels = num_levels
#         self.radius = radius
#         self.geo_volume_pyramid = []
#         self.init_corr_pyramid = []
#
#         # 这部分逻辑完全相同
#         init_corr = Combined_Geo_Encoding_Volume_Vertical.corr(init_fmap1, init_fmap2)
#         b, h, w, _, w2 = init_corr.shape
#         b, c, d, h, w = geo_volume.shape
#         geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1, d)
#         init_corr = init_corr.reshape(b * h * w, 1, 1, w2)
#
#         self.geo_volume_pyramid.append(geo_volume)
#         self.init_corr_pyramid.append(init_corr)
#
#         # 金字塔构建也相同
#         for i in range(self.num_levels - 1):
#             geo_volume = F.avg_pool2d(geo_volume, [1, 2], stride=[1, 2])
#             self.geo_volume_pyramid.append(geo_volume)
#         for i in range(self.num_levels - 1):
#             init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
#             self.init_corr_pyramid.append(init_corr)
#
#     def __call__(self, disp, coords):
#         r = self.radius
#         b, _, h, w = disp.shape
#         out_pyramid = []
#
#         for i in range(self.num_levels):
#             geo_volume = self.geo_volume_pyramid[i]
#
#             # 关键修改：从水平偏移改为垂直偏移
#             dy = torch.linspace(-r, r, 2 * r + 1)  # 垂直方向偏移
#             dy = dy.view(1, 1, 2 * r + 1, 1).to(disp.device)
#
#             #x0 = torch.zeros_like(dy)  # 水平方向固定为0
#             # y0 = dy + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i  # 垂直位移
#             # x0 = torch.zeros_like(y0)
#
#             x0 = dy + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i  # 垂直视差值映射到x采样
#             y0 = torch.zeros_like(x0)
#
#             disp_lvl = torch.cat([x0, y0], dim=-1)
#             geo_volume = bilinear_sampler(geo_volume, disp_lvl)
#             geo_volume = geo_volume.view(b, h, w, -1)  # 维度保持不变
#
#             init_corr = self.init_corr_pyramid[i]
#
#             init_x0 = coords.reshape(b * h * w, 1, 1, 1) / 2 ** i - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i + dy
#             init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
#
#             init_corr = bilinear_sampler(init_corr, init_coords_lvl)
#             init_corr = init_corr.view(b, h, w, -1)  # 维度保持不变
#
#             out_pyramid.append(geo_volume)
#             out_pyramid.append(init_corr)
#
#         out = torch.cat(out_pyramid, dim=-1)  # 拼接维度相同
#         return out.permute(0, 3, 1, 2).contiguous().float()  # 输出维度相同
#
#     @staticmethod
#     def corr(fmap1, fmap2):
#         # 完全相同，可以共用
#         B, D, H, W1 = fmap1.shape
#         _, _, _, W2 = fmap2.shape
#         fmap1 = fmap1.view(B, D, H, W1)
#         fmap2 = fmap2.view(B, D, H, W2)
#         corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
#         corr = corr.reshape(B, H, W1, 1, W2).contiguous()
#         return corr
#



class Combined_Geo_Encoding_Volume_2D_Reduction:
    def __init__(self, init_fmap1, init_fmap2, geo_volume_h, geo_volume_v,max_disp_h,max_disp_v,
                 num_levels=2, radius=4, reduction='max'):
        """
        Args:
            init_fmap1: 初始特征图1 [B, D, H, W]
            init_fmap2: 初始特征图2 [B, D, H, W]
            geo_volume_h: 水平几何编码体 [B, C, D_h, H, W]
            geo_volume_v: 垂直几何编码体 [B, C, D_v, H, W]
            num_levels: 金字塔层数
            radius: 搜索半径
            reduction: 降维方式 - 'max', 'mean', 'weighted', 'learned'
        """
        self.num_levels = num_levels
        self.radius = radius
        self.reduction = reduction
        self.geo_volume_h_pyramid = []
        self.geo_volume_v_pyramid = []
        self.init_corr_pyramid = []

        # 使用2D correlation
        init_corr = self.corr_2d(init_fmap1, init_fmap2,
                                 max_disp_h=max_disp_h,
                                 max_disp_v=max_disp_v)

        # init_corr = self.corr_2d_vectorized(init_fmap1, init_fmap2,
        #                          max_disp_h=max_disp_h,
        #                          max_disp_v=max_disp_v)

        b, h, w, max_disp_v, max_disp_h = init_corr.shape

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

    def __call__(self, disp_h, disp_v, coords):
        """
        执行2D搜索，然后通过降维提取水平和垂直代价
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

            # with autocast(enabled=False):  # 对于grid_sample，float32更稳定
            #     # 水平几何编码体采样
            #     geo_h_positions = torch.cat([search_positions[..., 0:1],
            #                                  torch.zeros_like(search_positions[..., 0:1])], dim=-1)
            #     geo_h_sampled = F.grid_sample(geo_volume_h, geo_h_positions.float(),
            #                                   mode='bilinear', padding_mode='zeros', align_corners=True)
            #
            #     # 垂直几何编码体采样
            #     geo_v_positions = torch.cat([torch.zeros_like(search_positions[..., 1:2]),
            #                                  search_positions[..., 1:2]], dim=-1)
            #     geo_v_sampled = F.grid_sample(geo_volume_v, geo_v_positions.float(),
            #                                   mode='bilinear', padding_mode='zeros', align_corners=True)
            #
            # geo_h_sampled = geo_h_sampled.view(b * h * w, -1, grid_shape[0], grid_shape[1])
            # geo_v_sampled = geo_v_sampled.view(b * h * w, -1, grid_shape[0], grid_shape[1])


            # ===== init_corr的2D采样 =====
            init_corr = self.init_corr_pyramid[i]

            coords_current = torch.stack([
                coords_x.reshape(b * h * w, 1, 1, 1) / 2 ** i,
                coords_y.reshape(b * h * w, 1, 1, 1) / 2 ** i
            ], dim=-1).squeeze(-2)

            init_search_positions = coords_current - disp_current + search_grid_flat

            init_corr_sampled = self.bilinear_sampler(init_corr, init_search_positions)
            # with autocast(enabled=False):
            #     init_corr_sampled = F.grid_sample(init_corr, init_search_positions.float(),
            #                                       mode='bilinear', padding_mode='zeros', align_corners=True)

            init_corr_sampled = init_corr_sampled.view(b * h * w, -1, grid_shape[0], grid_shape[1])

            # ===== 应用降维策略 =====
            # 对于水平代价：沿垂直方向(dim=2)降维
            geo_h_reduced = self._reduce_volume(geo_h_sampled, dim=2, direction='horizontal')
            init_corr_h_reduced = self._reduce_volume(init_corr_sampled, dim=2, direction='horizontal')

            # 对于垂直代价：沿水平方向(dim=3)降维
            geo_v_reduced = self._reduce_volume(geo_v_sampled, dim=3, direction='vertical')
            init_corr_v_reduced = self._reduce_volume(init_corr_sampled, dim=3, direction='vertical')

            # reshape回原始batch形状
            # geo_h_reduced = geo_h_reduced.view(b, h, w, -1)
            # init_corr_h_reduced = init_corr_h_reduced.view(b, h, w, -1)
            # geo_v_reduced = geo_v_reduced.view(b, h, w, -1)
            # init_corr_v_reduced = init_corr_v_reduced.view(b, h, w, -1)

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
        对2D搜索结果进行降维
        Args:
            volume: [B*H*W, C, grid_h, grid_w]
            dim: 降维的维度 (2 for horizontal, 3 for vertical)
            direction: 'horizontal' or 'vertical'
        Returns:
            reduced: [B*H*W, C, remaining_dim]
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
                # 垂直方向的权重（中心权重大，边缘权重小）
                weights = torch.exp(-torch.abs(torch.linspace(-r, r, 2*r+1, device=device)) / r)
                weights = weights / weights.sum()
                weights = weights.view(1, 1, -1, 1)
                reduced = (volume * weights).sum(dim=dim)
            else:
                # 水平方向的权重
                weights = torch.exp(-torch.abs(torch.linspace(-r, r, 2 * r + 1, device=device)) / r)
                weights = weights / weights.sum()
                weights = weights.view(1, 1, 1, -1)
                reduced = (volume * weights).sum(dim=dim)
        elif self.reduction == 'learned':
            # 使用可学习的权重
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
            # 先计算每个位置的相关性强度
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
    def corr_2d(fmap1, fmap2, max_disp_h, max_disp_v):
        """2D correlation计算"""
        B, D, H, W = fmap1.shape
        corr_volume = fmap1.new_zeros([B, H, W, max_disp_v, max_disp_h])

        for dy in range(max_disp_v):
            for dx in range(max_disp_h):
                if dy == 0 and dx == 0:
                    corr = torch.einsum('bdhw,bdhw->bhw', fmap1, fmap2)
                    corr_volume[:, :, :, dy, dx] = corr
                elif dy == 0:
                    valid_w = W - dx
                    corr = torch.einsum('bdhw,bdhw->bhw',
                                        fmap1[:, :, :, :valid_w],
                                        fmap2[:, :, :, dx:])
                    corr_volume[:, :, :valid_w, dy, dx] = corr
                elif dx == 0:
                    valid_h = H - dy
                    corr = torch.einsum('bdhw,bdhw->bhw',
                                        fmap1[:, :, :valid_h, :],
                                        fmap2[:, :, dy:, :])
                    corr_volume[:, :valid_h, :, dy, dx] = corr
                else:
                    valid_h = H - dy
                    valid_w = W - dx
                    corr = torch.einsum('bdhw,bdhw->bhw',
                                        fmap1[:, :, :valid_h, :valid_w],
                                        fmap2[:, :, dy:, dx:])
                    corr_volume[:, :valid_h, :valid_w, dy, dx] = corr

        return corr_volume.contiguous()

    @staticmethod
    def corr_2d_vectorized(fmap1, fmap2, max_disp_h, max_disp_v):
        """向量化的2D correlation计算"""
        B, D, H, W = fmap1.shape
        device = fmap1.device

        # 预分配输出张量
        corr_volume = torch.zeros([B, H, W, max_disp_v, max_disp_h],
                                  device=device, dtype=fmap1.dtype)

        # 批量计算所有位移
        for dy in range(max_disp_v):
            for dx in range(max_disp_h):
                if dy == 0 and dx == 0:
                    corr_volume[:, :, :, dy, dx] = (fmap1 * fmap2).sum(dim=1)
                elif dy == 0:
                    valid_w = W - dx
                    corr_volume[:, :, :valid_w, dy, dx] = (
                            fmap1[:, :, :, :valid_w] * fmap2[:, :, :, dx:]
                    ).sum(dim=1)
                elif dx == 0:
                    valid_h = H - dy
                    corr_volume[:, :valid_h, :, dy, dx] = (
                            fmap1[:, :, :valid_h, :] * fmap2[:, :, dy:, :]
                    ).sum(dim=1)
                else:
                    valid_h = H - dy
                    valid_w = W - dx
                    corr_volume[:, :valid_h, :valid_w, dy, dx] = (
                            fmap1[:, :, :valid_h, :valid_w] * fmap2[:, :, dy:, dx:]
                    ).sum(dim=1)

        return corr_volume

    @staticmethod
    def corr_2d_fast(fmap1, fmap2, max_disp_h, max_disp_v):
        B, D, H, W = fmap1.shape
        device = fmap1.device

        # 预分配输出
        corr_volume = torch.zeros([B, H, W, max_disp_v, max_disp_h],
                                  device=device, dtype=fmap1.dtype)

        # 重塑为矩阵乘法友好的形式
        fmap1_flat = fmap1.reshape(B, D, -1).transpose(1, 2)  # [B, H*W, D]

        # 批量处理每个位移
        for dy in range(max_disp_v):
            for dx in range(max_disp_h):
                if dy == 0 and dx == 0:
                    fmap2_flat = fmap2.reshape(B, D, -1).transpose(1, 2)  # [B, H*W, D]
                    corr = torch.bmm(fmap1_flat, fmap2_flat.transpose(1, 2)).diagonal(dim1=1, dim2=2)
                    corr_volume[:, :, :, dy, dx] = corr.reshape(B, H, W)
                elif dy == 0:
                    valid_w = W - dx
                    fmap2_shifted = fmap2[:, :, :, dx:].contiguous()
                    fmap2_flat = fmap2_shifted.reshape(B, D, -1).transpose(1, 2)
                    fmap1_part = fmap1[:, :, :, :valid_w].reshape(B, D, -1).transpose(1, 2)
                    corr = (fmap1_part * fmap2_flat).sum(dim=2)
                    corr_volume[:, :, :valid_w, dy, dx] = corr.reshape(B, H, valid_w)
                elif dx == 0:
                    valid_h = H - dy
                    fmap2_shifted = fmap2[:, :, dy:, :].contiguous()
                    fmap2_flat = fmap2_shifted.reshape(B, D, -1).transpose(1, 2)
                    fmap1_part = fmap1[:, :, :valid_h, :].reshape(B, D, -1).transpose(1, 2)
                    corr = (fmap1_part * fmap2_flat).sum(dim=2)
                    corr_volume[:, :valid_h, :, dy, dx] = corr.reshape(B, valid_h, W)
                else:
                    valid_h = H - dy
                    valid_w = W - dx
                    fmap2_shifted = fmap2[:, :, dy:, dx:].contiguous()
                    fmap2_flat = fmap2_shifted.reshape(B, D, -1).transpose(1, 2)
                    fmap1_part = fmap1[:, :, :valid_h, :valid_w].reshape(B, D, -1).transpose(1, 2)
                    corr = (fmap1_part * fmap2_flat).sum(dim=2)
                    corr_volume[:, :valid_h, :valid_w, dy, dx] = corr.reshape(B, valid_h, valid_w)

        return corr_volume


    #@staticmethod
    # def bilinear_sampler(img, coords):
    #     """双线性采样"""
    #     return F.grid_sample(img, coords, mode='bilinear',
    #                          padding_mode='zeros', align_corners=True)





