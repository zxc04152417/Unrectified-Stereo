import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_gwc_volume_vertical(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])

    for i in range(maxdisp):
        if i > 0:
            # 垂直方向移位：在高度维度上移位
            volume[:, :, i, i:, :] = groupwise_correlation(
                refimg_fea[:, :, i:, :],  # 参考图像特征的下边部分
                targetimg_fea[:, :, :-i, :],  # 目标图像特征的上边部分
                num_groups
            )
        else:
            # i=0时，不需要移位
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)

    volume = volume.contiguous()
    return volume


def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost

def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume



def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def build_gwc_volume_signed(refimg_fea, targetimg_fea, maxdisp_pos, maxdisp_neg, num_groups):
    """
    水平带符号代价体:
      dx ∈ [-maxdisp_neg, ..., -1, 0, 1, ..., maxdisp_pos-1]
    输出体深度 D = maxdisp_neg + maxdisp_pos  (包含 0 平面)
    仍返回形状: [B, G, D, H, W]，可直接接后续 3D 卷积。

    说明：
      - 和你原来的正向实现严格对称：正 dx 用右图向左裁切，负 dx 用右图向右裁切。
      - 0 平面仍是全幅相关。
    """
    B, C, H, W = refimg_fea.shape
    D = maxdisp_neg + maxdisp_pos
    volume = refimg_fea.new_zeros([B, num_groups, D, H, W])

    def plane_idx(dx):  # dx ∈ [-maxdisp_neg, ..., maxdisp_pos-1]
        return dx + maxdisp_neg

    # 负 dx：右图向右裁切
    for a in range(maxdisp_neg, 0, -1):     # a = 1..maxdisp_neg
        dx = -a
        # 有效宽度
        We = W - a
        if We <= 0:
            continue
        # ref: [:, :, :, :W-a]   tgt: [:, :, :, a:]
        cost = groupwise_correlation(
            refimg_fea[:, :, :, :We].contiguous(),
            targetimg_fea[:, :, :, a:].contiguous(),
            num_groups
        )
        volume[:, :, plane_idx(dx), :, :We] = cost

    # dx = 0：全幅
    volume[:, :, plane_idx(0), :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)

    # 正 dx：右图向左裁切（与原实现一致）
    for dx in range(1, maxdisp_pos):
        We = W - dx
        if We <= 0:
            continue
        cost = groupwise_correlation(
            refimg_fea[:, :, :, dx:].contiguous(),
            targetimg_fea[:, :, :, :We].contiguous(),
            num_groups
        )
        volume[:, :, plane_idx(dx), :, dx:] = cost

    return volume.contiguous()

def build_gwc_volume_vertical_signed(refimg_fea, targetimg_fea, maxdisp, num_groups):
    """
    垂直方向带符号的 GWC 代价体:
      dy ∈ [-maxdisp, ..., 0, ..., +maxdisp]
    输出: [B, num_groups, D, H, W]，D = 2*maxdisp + 1

    说明：
      - dy > 0 表示右图相对左图向“上/下”移动（与数据约定一致即可）。
      - 索引写回时，只在有效重叠区域写入，其他位置保持 0。
    """
    B, C, H, W = refimg_fea.shape
    D = 2 * maxdisp + 1
    volume = refimg_fea.new_zeros([B, num_groups, D, H, W])

    def plane_idx(dy: int) -> int:
        # 将 [-maxdisp, ..., 0, ..., +maxdisp] 映射到 [0, ..., D-1]
        return dy + maxdisp

    for dy in range(-maxdisp, maxdisp + 1):
        He = H - abs(dy)
        if He <= 0:
            continue

        # 参考/目标图在高度维的有效切片
        ys_ref = max(dy, 0)    # dy>0: 参考向下裁切；dy<0: 参考从0开始
        ys_tgt = max(-dy, 0)   # 与参考相反

        ref_patch = refimg_fea[:, :, ys_ref:ys_ref + He, :]     # [B,C,He,W]
        tgt_patch = targetimg_fea[:, :, ys_tgt:ys_tgt + He, :]  # [B,C,He,W]

        cost = groupwise_correlation(ref_patch.contiguous(),
                                     tgt_patch.contiguous(),
                                     num_groups)                # [B,G,He,W]

        # 写回到对应的 disparity 平面，只覆盖有效区域
        volume[:, :, plane_idx(dy), ys_ref:ys_ref + He, :] = cost

    return volume.contiguous()


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def disparity_regression1(
    prob: torch.Tensor,
    maxdisp: int = None,
    *,
    maxdisp_pos: int = None,
    maxdisp_neg: int = 0,
    disp_values: torch.Tensor = None,
):
    """
    统一的 soft-argmax 视差回归函数，支持正负视差，并保持与旧接口的兼容。

    参数:
        prob: [B, D, H, W]  已在 D 维做完 softmax 的概率体
        maxdisp: 旧版参数（仅正视差），等价于 maxdisp_pos；当 maxdisp_neg=0 时完全兼容旧行为
        maxdisp_pos: 正向最大视差个数（不含符号平面数）；若未提供则回退到 maxdisp
        maxdisp_neg: 负向最大视差个数（不含 0 平面数）；为 0 时表示不支持负视差（与旧版一致）
        disp_values: 可选，显式传入视差取值表 [D] 或 [1,D,1,1]，优先级最高

    返回:
        disp: [B, 1, H, W]  期望视差

    兼容性说明:
        - 旧用法: disparity_regression(prob, maxdisp=M)
          等价于 disparity_regression(prob, maxdisp_pos=M, maxdisp_neg=0)
          取值表为 [0, 1, ..., M-1]，与旧版完全一致。
        - 新用法(带负视差): disparity_regression(prob, maxdisp_pos=Px, maxdisp_neg=Ny)
          取值表为 [-Ny, ..., -1, 0, 1, ..., Px-1]。
        - 若传入 disp_values，则直接用它（用于与自定义/填充后的 D 完全对齐）。
    """
    assert prob.dim() == 4, f"prob shape should be [B, D, H, W], got {prob.shape}"
    B, D, H, W = prob.shape
    device, dtype = prob.device, prob.dtype

    # 1) 若显式提供了取值表，则直接使用
    if disp_values is not None:
        if disp_values.dim() == 1:
            vals = disp_values.view(1, -1, 1, 1).to(device=device, dtype=dtype)
        elif disp_values.dim() == 4:
            # 允许传 [1, D, 1, 1] 形式
            vals = disp_values.to(device=device, dtype=dtype)
        else:
            raise ValueError("disp_values must be shape [D] or [1,D,1,1].")
        assert vals.shape[1] == D, f"disp_values length {vals.shape[1]} != D={D}"
        return (prob * vals).sum(dim=1, keepdim=True)

    # 2) 兼容旧接口：没给 maxdisp_pos 就用 maxdisp；没给 maxdisp 就报错
    if maxdisp_pos is None:
        if maxdisp is None:
            raise ValueError("Provide either maxdisp (old API) or maxdisp_pos/maxdisp_neg.")
        maxdisp_pos = maxdisp  # 兼容旧版

    # 3) 构造带符号/不带符号的取值表
    if maxdisp_neg > 0:
        # 带符号: [-maxdisp_neg, ..., -1, 0, 1, ..., maxdisp_pos-1]
        vals_1d = torch.arange(-maxdisp_neg, maxdisp_pos, device=device, dtype=dtype)
    else:
        # 旧版行为: [0, 1, ..., maxdisp_pos-1]
        vals_1d = torch.arange(0, maxdisp_pos, device=device, dtype=dtype)

    # 4) 与 D 对齐的健壮性检查
    if vals_1d.numel() != D:
        raise ValueError(
            f"Length of disparity values ({vals_1d.numel()}) != D ({D}). "
            "Check that your cost volume depth matches (neg + pos)."
        )

    vals = vals_1d.view(1, D, 1, 1)  # [1,D,1,1]
    disp = (prob * vals).sum(dim=1, keepdim=True)  # [B,1,H,W]
    return disp






class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv

def context_upsample(disp_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape
        
    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)
        
    return disp

class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples):

        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                    one_hot_filter,padding=0)
                                                    
        return aggregated_disparity_samples
        

class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)


        return prob_volume_propa


def build_2d_gwc_volume(refimg_fea, targetimg_fea, maxdisp_h, maxdisp_v, num_groups):
    """
    构建二维搜索空间的代价体
    Args:
        refimg_fea: 参考图像特征 [B, C, H, W]
        targetimg_fea: 目标图像特征 [B, C, H, W]
        maxdisp_h: 水平方向最大视差
        maxdisp_v: 垂直方向最大视差
        num_groups: 分组数
    Returns:
        volume_2d: 二维代价体 [B, num_groups, maxdisp_v, maxdisp_h, H, W]
    """
    B, C, H, W = refimg_fea.shape
    volume_2d = refimg_fea.new_zeros([B, num_groups, maxdisp_v, maxdisp_h, H, W])    #这个地方在2D代价体里面就是先垂直再水平了

    for dy in range(maxdisp_v):
        for dx in range(maxdisp_h):
            if dy == 0 and dx == 0:
                # 无位移
                volume_2d[:, :, dy, dx, :, :] = groupwise_correlation(
                    refimg_fea, targetimg_fea, num_groups
                )
            elif dy == 0:
                # 仅水平位移
                volume_2d[:, :, dy, dx, :, dx:] = groupwise_correlation(
                    refimg_fea[:, :, :, dx:],
                    targetimg_fea[:, :, :, :-dx],
                    num_groups
                )
            elif dx == 0:
                # 仅垂直位移
                volume_2d[:, :, dy, dx, dy:, :] = groupwise_correlation(
                    refimg_fea[:, :, dy:, :],
                    targetimg_fea[:, :, :-dy, :],
                    num_groups
                )
            else:
                # 水平和垂直同时位移
                volume_2d[:, :, dy, dx, dy:, dx:] = groupwise_correlation(
                    refimg_fea[:, :, dy:, dx:],
                    targetimg_fea[:, :, :-dy, :-dx],
                    num_groups
                )

    return volume_2d.contiguous()



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
        # 对垂直维度取最大值得到水平代价体
        volume_h, _ = torch.max(volume_2d, dim=2)  # [B, num_groups, maxdisp_h, H, W]
        # 对水平维度取最大值得到垂直代价体
        volume_v, _ = torch.max(volume_2d, dim=3)  # [B, num_groups, maxdisp_v, H, W]
    elif reduction == 'mean':
        # 对垂直维度取平均值得到水平代价体
        volume_h = torch.mean(volume_2d, dim=2)  # [B, num_groups, maxdisp_h, H, W]
        # 对水平维度取平均值得到垂直代价体
        volume_v = torch.mean(volume_2d, dim=3)  # [B, num_groups, maxdisp_v, H, W]
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")

    return volume_h, volume_v


def build_gwc_volumes_with_2d_search(refimg_fea, targetimg_fea, maxdisp_h, maxdisp_v, num_groups, reduction='max'):
    """
    使用二维搜索构建水平和垂直代价体
    Args:
        refimg_fea: 参考图像特征
        targetimg_fea: 目标图像特征
        maxdisp_h: 水平最大视差
        maxdisp_v: 垂直最大视差
        num_groups: 分组数
        reduction: 从2D代价体提取1D代价体的方式
    Returns:
        volume_h: 水平代价体 [B, num_groups, maxdisp_h, H, W]
        volume_v: 垂直代价体 [B, num_groups, maxdisp_v, H, W]
    """
    # 构建二维代价体
    volume_2d = build_2d_gwc_volume(refimg_fea, targetimg_fea, maxdisp_h, maxdisp_v, num_groups)

    # 提取水平和垂直代价体
    volume_h, volume_v = extract_horizontal_vertical_volumes(volume_2d, reduction)

    return volume_h, volume_v


