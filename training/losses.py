import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np

class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))


class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


class Adversarial_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean((fake_outputs - 1) ** 2 / 2)


class Discriminate_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)


# ==========================================================
# B3: Temporal + Bezier (flow-warp temporal, flow-trajectory bezier)
# ==========================================================

def ramp(step, start, end, max_val):
    if step < start:
        return 0.0
    if step >= end:
        return float(max_val)
    alpha = (step - start) / float(end - start)
    return float(max_val) * float(alpha)


def flow_warp(x, flow):
    """
    x:    (N,C,H,W)
    flow: (N,2,H,W)  (u,v) pixel displacement
    """
    N, C, H, W = x.size()
    yy, xx = torch.meshgrid(
        torch.arange(0, H, device=x.device),
        torch.arange(0, W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).repeat(N, 1, 1, 1)  # (N,2,H,W)

    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=3)  # (N,H,W,2)

    return torch.nn.functional.grid_sample(
        x, vgrid_norm, mode='bilinear', padding_mode='border', align_corners=True
    )


def temporal_consistency_loss(pred_next, frame_cur, flow_cur_to_next):
    """
    pred_next: predicted frame_5 (N,3,H,W)
    frame_cur: real frame_4 (N,3,H,W)
    flow_cur_to_next: flow frame_4 -> frame_5 (N,2,H,W)
    """
    pred_back = flow_warp(pred_next, -flow_cur_to_next)  # approx inverse warp
    return torch.nn.functional.l1_loss(pred_back, frame_cur)


def _sample_flow_at(flow, coords_xy):
    """
    flow: (N,2,H,W)
    coords_xy: (N,2,H,W) absolute pixel coords (x,y)
    returns: (N,2,H,W)
    """
    N, _, H, W = flow.shape
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (N,H,W,2)
    return torch.nn.functional.grid_sample(
        flow, grid, mode='bilinear', padding_mode='border', align_corners=True
    )


def _bezier_point(P0, C1, C2, P3, t: float):
    one = 1.0 - t
    return (one**3)*P0 + 3*(one**2)*t*C1 + 3*one*(t**2)*C2 + (t**3)*P3


def bezier_trajectory_loss(flow_12, flow_23, flow_34, flow_45):
    """
    5 frame zinciri: 1->2->3->4->5
    P0..P4 trajektori oluştur, iki cubic segment ile (P0->P3) ve (P1->P4)
    Bezier ara noktaları discrete noktalarla uyumlu olsun diye cezalandır.
    """
    N, _, H, W = flow_12.shape
    yy, xx = torch.meshgrid(
        torch.arange(0, H, device=flow_12.device),
        torch.arange(0, W, device=flow_12.device),
        indexing='ij'
    )
    P0 = torch.stack([xx, yy], dim=0).float().unsqueeze(0).repeat(N, 1, 1, 1)  # (N,2,H,W)
    P1 = P0 + flow_12

    f23 = _sample_flow_at(flow_23, P1)
    P2 = P1 + f23

    f34 = _sample_flow_at(flow_34, P2)
    P3 = P2 + f34

    f45 = _sample_flow_at(flow_45, P3)
    P4 = P3 + f45

    # Segment A: P0->P3, C1=P1, C2=P2
    BA13 = _bezier_point(P0, P1, P2, P3, 1.0/3.0)
    BA23 = _bezier_point(P0, P1, P2, P3, 2.0/3.0)
    lossA = torch.mean(torch.abs(BA13 - P1)) + torch.mean(torch.abs(BA23 - P2))

    # Segment B: P1->P4, C1=P2, C2=P3
    BB13 = _bezier_point(P1, P2, P3, P4, 1.0/3.0)
    BB23 = _bezier_point(P1, P2, P3, P4, 2.0/3.0)
    lossB = torch.mean(torch.abs(BB13 - P2)) + torch.mean(torch.abs(BB23 - P3))

    return lossA + lossB
