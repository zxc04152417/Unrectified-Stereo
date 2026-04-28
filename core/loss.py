import torch
import torch.nn.functional as F

def _safe_mean(x, m):
    # x, m: [B,1,H,W]（或可 broadcast 到这个形状）
    m = m.bool()
    if m.any():
        return x[m].mean()
    else:
        return (x*0).sum()

def _valid_mask(gt, base_valid, max_disp):
    # gt: [B,1,H,W], base_valid: [B,H,W] in {0,1}
    if base_valid is None:
        base_valid = torch.ones(gt.shape[0], gt.shape[2], gt.shape[3],
                                device=gt.device, dtype=torch.bool)
    else:
        base_valid = base_valid > 0.5

    finite = torch.isfinite(gt).squeeze(1) & (~torch.isnan(gt).squeeze(1))
    mag_ok = gt.abs().squeeze(1) < max_disp
    return (base_valid & finite & mag_ok).unsqueeze(1)   # [B,1,H,W]

def _seq_weights(n, gamma=0.9):
    if n <= 1: return [1.0]
    return [gamma ** (n - i - 1) for i in range(n)]

def _seq_branch_loss(preds, gt, valid, seq_w, init_pred=None, init_w=0.2, robust='l1'):
    # preds: list of [B,1,H,W]
    # gt:    [B,1,H,W], valid: [B,1,H,W] (bool)
    loss = gt.new_tensor(0.0)
    m = valid.bool() & torch.isfinite(gt) & (~torch.isnan(gt))

    def _robust_l1(a, b):
        if robust == 'sl1':
            return F.smooth_l1_loss(a, b, reduction='none')
        else:
            return (a - b).abs()

    # init term (optional)
    if init_pred is not None and init_w > 0:
        ip = torch.where(torch.isfinite(init_pred), init_pred, torch.zeros_like(init_pred))
        loss = loss + init_w * _safe_mean(_robust_l1(ip, gt), m)

    # sequence terms
    for wi, pi in zip(seq_w, preds):
        pi = torch.where(torch.isfinite(pi), pi, torch.zeros_like(pi))
        loss = loss + wi * _safe_mean(_robust_l1(pi, gt), m)

    return loss

def _epe_metrics(pred, gt, valid):
    # pred, gt: [B,1,H,W], valid: [B,1,H,W]
    err = (pred - gt).pow(2).sum(dim=1).sqrt()  # [B,H,W]（1通道退化为 |.|）
    m = valid.view(-1)
    e = err.view(-1)[m]
    if e.numel() == 0:
        z = pred.new_tensor(0.0)
        return {'epe': z, '1px': z, '3px': z, '5px': z}
    return {
        'epe':  e.mean(),
        '1px': (e < 1).float().mean(),
        '3px': (e < 3).float().mean(),
        '5px': (e < 5).float().mean(),
    }

def sequence_loss_stable(
    # 水平
    disp_preds_h, disp_init_h, disp_gt_h, valid_h,
    # 垂直
    disp_preds_v, disp_init_v, disp_gt_v, valid_v,
    # 矫正一致性（由 (h,v) 预测“反变换”得到的 rectified 预测；以及 rectified GT/valid）
    disp_preds_rect, disp_init_rect, disp_gt_rect, valid_rect,
    *,
    loss_gamma=0.9, max_disp_h=192, max_disp_v=64, robust='l1',
    init_weight=0.2,
    lambda_consistency=0.2,    # 矫正一致性的小权重（建议 0.1~0.3）
    adapt_consistency=True     # 按有效像素比例自适应缩放一致性权重
):
    """
    设计意图：
      - 主监督：水平 + 垂直
      - 矫正只做“一致性”项，权重小，避免和主监督重复/冲突
      - 每路都走自己的掩码与幅值阈值；所有 mean 都做空保护
    约定：
      - 所有 disp_* 与 gt： [B,1,H,W]
      - 所有 valid_*：      [B,H,W] in {0,1}
      - disp_preds_rect/disp_init_rect 是由 (h,v) 经过你那套反变换得到的“rectified 预测”
    """

    n = len(disp_preds_h)
    assert n >= 1 and len(disp_preds_v) == n and len(disp_preds_rect) == n

    # 各自掩码
    vm_h = _valid_mask(disp_gt_h,   valid_h,   max_disp_h)  # [B,1,H,W]
    vm_v = _valid_mask(disp_gt_v,   valid_v,   max_disp_v)  # 垂直通常范围更小
    vm_r = _valid_mask(disp_gt_rect, valid_rect, max_disp_h)

    # 序列权重
    sw = _seq_weights(n, gamma=loss_gamma)

    # 主监督：水平 + 垂直
    loss_h = _seq_branch_loss(disp_preds_h, disp_gt_h, vm_h, sw,
                              init_pred=disp_init_h, init_w=init_weight, robust=robust)
    loss_v = _seq_branch_loss(disp_preds_v, disp_gt_v, vm_v, sw,
                              init_pred=disp_init_v, init_w=init_weight, robust=robust)

    # 矫正一致性（小权重）：只监督 rectified 空间的误差，避免与 h/v 重复监督
    loss_r = _seq_branch_loss(disp_preds_rect, disp_gt_rect, vm_r, sw,
                              init_pred=disp_init_rect, init_w=init_weight*0.5, robust=robust)

    # 自适应缩放一致性权重（按有效像素占比，防止掩码差异导致不平衡）
    if adapt_consistency:
        nh = vm_h.sum().clamp(min=1).float()
        nv = vm_v.sum().clamp(min=1).float()
        nr = vm_r.sum().clamp(min=1).float()
        # 参照：(nh+nv)/2 与 nr 的比例
        ratio = ( (nh + nv) * 0.5 ) / nr
        ratio = torch.sqrt(ratio)              # 温和一点
        ratio = torch.clamp(ratio, 0.5, 2.0)   # 裁剪
        lam_r = float(lambda_consistency * ratio)
    else:
        lam_r = lambda_consistency

    total = loss_h + loss_v + lam_r * loss_r

    # 指标（看最后一帧）
    m_h = _epe_metrics(disp_preds_h[-1],  disp_gt_h,   vm_h)
    m_v = _epe_metrics(disp_preds_v[-1],  disp_gt_v,   vm_v)
    m_r = _epe_metrics(disp_preds_rect[-1], disp_gt_rect, vm_r)

    metrics = {
        'total_loss': total,
        'loss_h': loss_h, 'loss_v': loss_v, 'loss_rect': loss_r,
        'lam_rect': torch.tensor(lam_r, device=total.device),
        'epe_h': m_h['epe'], '1px_h': m_h['1px'], '3px_h': m_h['3px'], '5px_h': m_h['5px'],
        'epe_v': m_v['epe'], '1px_v': m_v['1px'], '3px_v': m_v['3px'], '5px_v': m_v['5px'],
        'epe_rect': m_r['epe'], '1px_rect': m_r['1px'], '3px_rect': m_r['3px'], '5px_rect': m_r['5px'],
    }
    return total, metrics
