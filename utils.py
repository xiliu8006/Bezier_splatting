import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
import math
import copy
import cv2
import numpy.random as npr
import matplotlib.pyplot as plt
from skimage.color import label2rgb

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

# #code modified from https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization/blob/679e1d16c5809367f2d2db3e403a8548c5419258/LIVE/xing_loss.py#L22C15-L22C21
# def compute_sine_theta(s1, s2):  #s1 and s2 aret two segments to be uswed
#     #s1, s2 (2, 2)
#     v1 = s1[1,:] - s1[0, :]
#     v2 = s2[1,:] - s2[0, :]
#     #print(v1, v2)
#     sine_theta = ( v1[0] * v2[1] - v1[1] * v2[0] ) / (torch.norm(v1) * torch.norm(v2))
#     return sine_theta

# def xing_loss(control_points, scale=1e-3):
def inverse_sigmoid(x):
    return torch.log(x/(1-x))
    
def curvature_regularization(sampled_points, W, H, weight=10.0):
    # print("sample points shape: ", sampled_points.shape)
    scaled_x = sampled_points[:, :, 0:1] * W * 0.5 + W * 0.5
    scaled_y = sampled_points[:, :, 1:2] * H * 0.5 + H * 0.5
    sampled_point = torch.cat([scaled_x, scaled_y], dim=2)
    # print("loss curvature: ", sampled_point.max(), sampled_point.min(), W, H, sampled_point.shape)
    second_deriv = sampled_point[:, 2:, :] - 2 * sampled_point[:, 1:-1, :] + sampled_point[:, :-2, :]
    curvature_sq = torch.sum(second_deriv ** 2, dim=-1)  # shape: (N, H-2)
    loss_curvature = weight * curvature_sq.mean()
    # print("loss curvature: ", loss_curvature)
    return loss_curvature

def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R

def xing_loss(x_tensor, scale=1e-3):
    """
    x_tensor: a tensor of shape (N, 10, 2), where each row is a curve with 10 points (not closed).
              Since (10 - 1)=9 segments, and 9 % 3 == 0, we can group every 3 consecutive segments.
    The function computes a loss that encourages each group of 3 consecutive segments to have
    consistent turning direction, and it is computed in parallel for all N curves.
    """
    N, n_points, _ = x_tensor.shape
    num_segments = n_points - 1

    remainder = num_segments % 3
    if remainder != 0:
        num_missing = 3 - remainder  # e.g. remainder=1 -> need 2 points
        # 取前几个点补在末尾（保持曲线首尾连续）
        to_add = x_tensor[:, 1:1 + num_missing, :]  # 或 0:1 if你只想补P0
        x_tensor = torch.cat([x_tensor, to_add], dim=1)
        n_points = x_tensor.shape[1]
        num_segments = n_points - 1

    assert num_segments % 3 == 0
    num_groups = num_segments // 3

    # print("x_tensor shape is: ", x_tensor.shape)
    segments = torch.stack([x_tensor[:, :-1, :], x_tensor[:, 1:, :]], dim=2)
    segments = segments.view(N, num_groups, 3, 2, 2)

    # For each group, extract the three segments: cs1, cs2, cs3
    # Each has shape: (N, num_groups, 2, 2)
    cs1 = segments[:, :, 0, :, :]
    cs2 = segments[:, :, 1, :, :]
    cs3 = segments[:, :, 2, :, :]

    # Compute direction vectors for each segment.
    # For cs1: v1 = cs1[1] - cs1[0], shape: (N, num_groups, 2)
    v1 = cs1[:, :, 1, :] - cs1[:, :, 0, :]
    v2 = cs2[:, :, 1, :] - cs2[:, :, 0, :]
    v3 = cs3[:, :, 1, :] - cs3[:, :, 0, :]

    # Compute sine of angles:
    # sine_theta between cs1 and cs2:
    cross_12 = v1[..., 0]*v2[..., 1] - v1[..., 1]*v2[..., 0]
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)
    sine_theta_12 = cross_12 / (norm1 * norm2 + 1e-8)

    # sine_theta between cs1 and cs3:
    cross_13 = v1[..., 0]*v3[..., 1] - v1[..., 1]*v3[..., 0]
    norm3 = torch.norm(v3, dim=-1)
    sine_theta_13 = cross_13 / (norm1 * norm3 + 1e-8)

    # Determine turning direction from cs1 to cs2.
    # If sine_theta_12 >= 0, direct = 1 (e.g. counter-clockwise); otherwise 0.
    direct = (sine_theta_12 >= 0).float()
    opst = 1 - direct

    # Penalize inconsistency in turning:
    # If direct==1, expect sine_theta_13 to be non-negative, so penalize negative values.
    # If direct==0, expect sine_theta_13 to be non-positive, so penalize positive values.
    loss_groups = direct * torch.relu(-sine_theta_13) + opst * torch.relu(sine_theta_13)
    
    # Average the loss over groups and then over all curves.
    seg_loss = loss_groups.mean(dim=1)  # shape: (N,)
    loss = seg_loss.mean() * scale
    return loss

def curvature_loss(paths: torch.Tensor, H: int, angle_thresh_deg=60.0) -> torch.Tensor:
    """
    计算由 paths[:, 0] 和 flip(paths[:, 1]) 拼接后的路径，在每隔 H 处连接点的曲率损失。

    参数:
        paths: Tensor [N, 2, M, 2]，两条拼接 Bézier 曲线段采样点
        H: 每段 Bézier 采样点数（用于确定连接点位置）

    返回:
        标量 curvature loss
    """
    assert paths.dim() == 4 and paths.shape[1] == 2, "Expect input shape [N, 2, M, 2]"
    N, _, M, _ = paths.shape
    path1 = paths[:, 0, :, :]              # [N, M, 2]
    path2 = torch.flip(paths[:, 1, :, :], dims=[1])  # [N, M, 2]
    full_path = torch.cat([path1, path2], dim=1)     # [N, 2M, 2]
    total_len = full_path.shape[1]

    # 拼接点索引
    indices = torch.arange(0, total_len, H, device=paths.device)  # [0, H, 2H, ...]
    # roll 相邻点
    prev = torch.roll(full_path, 5, dims=1)[:, indices, :]  # [N, K, 2]
    curr = full_path[:, indices, :]
    nex = torch.roll(full_path, -5, dims=1)[:, indices, :]

    # --- Curvature ---
    second_diff = prev - 2 * curr + nex           # [N, K, 2]
    curvature = second_diff.pow(2).sum(dim=-1)    # [N, K]

    # --- Angle ---
    v1 = F.normalize(prev - curr, dim=-1)   # [N, K, 2]
    v2 = F.normalize(nex - curr, dim=-1)
    cos_theta = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_theta)           # [N, K]

    # --- Mask ---
    angle_thresh_rad = math.radians(angle_thresh_deg)
    mask = (angle < angle_thresh_rad).float()  # [N, K]

    # --- Apply curvature only if direction is aligned ---
    loss = (curvature * mask).mean()
    return loss

def boundary_loss_on_joints(points: torch.Tensor, degree: int, bound: float = 1.0) -> torch.Tensor:
    """
    只对 Bézier 曲线的连接点（每隔 degree 个点）做边界约束。

    参数:
        points: Tensor of shape (N, M, 2)
        degree: Bézier 曲线的 degree（如 3）
        bound: 边界，通常为 1.0 表示 [-1, 1]

    返回:
        标量 loss
    """
    assert points.dim() == 3 and points.shape[-1] == 2, "Expect shape (N, M, 2)"
    N, M, _ = points.shape

    # 连接点索引：0, degree, 2*degree, ...
    joint_indices = torch.arange(0, M, degree, device=points.device)  # shape: [K]
    # print(points.shape, joint_indices， points.shape.max(), points.shape.min())

    # 取出所有连接点
    joints = points[:, joint_indices, :]  # shape: (N, K, 2)

    # 超出 [-1, 1] 的部分
    over = torch.relu(joints - bound)
    under = torch.relu(-bound - joints)
    excess = over + under  # shape: (N, K, 2)

    return excess.mean()


# def curvature_loss(paths: torch.Tensor, angle_thresh_deg=60.0) -> torch.Tensor:
#     """
#     Compute curvature loss for either:
#         - paths: (N, M, 2) → single curve per sample
#         - paths: (N, 2, M, 2) → two sides per sample, side 1 is flipped before computing

#     Returns:
#         Scalar tensor: mean curvature loss
#     """
#     if paths.dim() == 3:
#         # Standard case: (N, M, 2)
#         N, M, _ = paths.shape
#         if M < 3:
#             return torch.tensor(0.0, device=paths.device)
#         second_diff = paths[:, :-2, :] - 2 * paths[:, 1:-1, :] + paths[:, 2:, :]
#         squared_norms = second_diff.pow(2).sum(dim=-1)  # (N, M-2)
#         return squared_norms.mean()

#     elif paths.dim() == 4:
#         path1 = paths[:,0, :, :]
#         path2 = paths[:,1, :, :]

#         P_start = path1[:, 0, :]  # (N, 2)
#         P_start_prev = path1[:, 4, :]
#         P_start_next = path2[:, 4, :]
#         # print("point: ", P_start, P_start_prev, P_start_next)
#         P_end = path2[:, -1, :]
#         P_end_prev = path2[:, -4, :]
#         P_end_next = path1[:, -4, :]

#         # curvature = ||P_{i-1} - 2*P_i + P_{i+1}||^2
#         eps = 1e-6  # 防止除以 0

#         # --- Start curvature ---
#         numerator_start = (P_start_prev - 2 * P_start + P_start_next).pow(2).sum(dim=-1)
#         denominator_start = (P_start_prev - P_start).pow(2).sum(dim=-1) + (P_start_next - P_start).pow(2).sum(dim=-1)
#         curv_start = numerator_start / (denominator_start + eps)

#         # --- End curvature ---
#         numerator_end = (P_end_prev - 2 * P_end + P_end_next).pow(2).sum(dim=-1)
#         denominator_end = (P_end_prev - P_end).pow(2).sum(dim=-1) + (P_end_next - P_end).pow(2).sum(dim=-1)
#         curv_end = numerator_end / (denominator_end + eps)
 

#         # --- Angle ---
#         vec_start_1 = F.normalize(P_start_prev - P_start, dim=-1)
#         vec_start_2 = F.normalize(P_start_next - P_start, dim=-1)
#         cos_angle_start = (vec_start_1 * vec_start_2).sum(dim=-1).clamp(-1.0, 1.0)
#         angle_start = torch.acos(cos_angle_start)  # (N,)

#         vec_end_1 = F.normalize(P_end_prev - P_end, dim=-1)
#         vec_end_2 = F.normalize(P_end_next - P_end, dim=-1)
#         cos_angle_end = (vec_end_1 * vec_end_2).sum(dim=-1).clamp(-1.0, 1.0)
#         angle_end = torch.acos(cos_angle_end)  # (N,)

#         # --- Threshold filtering ---
#         thresh_rad = math.radians(angle_thresh_deg)
#         mask_start = (angle_start < thresh_rad).float()
#         mask_end = (angle_end < thresh_rad).float()
#         # --- Apply masks ---
#         # loss = (curv_start * mask_start + curv_end * mask_end).mean() * 1e-2
#         loss = (curv_start + curv_end).mean()
#         # print("curves loss: ", loss, angle_start, angle_end, curv_start, curv_end)
#     return loss


# def bezier_shape_regularizer(control_points, lambda_proj=1e-2, lambda_center=0.1):
#     """
#     control_points: (N, 8, 2)
#     Returns: scalar loss
#     """
#     N = control_points.size(0)
#     P0 = control_points[:, 0, :]  # (N, 2)
#     P4 = control_points[:, 4, :]  # (N, 2)
#     center = 0.5 * (P0 + P4)      # (N, 2)

#     mids = control_points[:, [1, 2, 3, 7, 6, 5], :]  # (N, 6, 2)
#     P0_exp = P0[:, None, :]       # (N, 1, 2)
#     # center_exp = center[:, None, :]

#     v = P4 - P0                   # (N, 2)
#     v_norm_sq = (v ** 2).sum(dim=1, keepdim=True) + 1e-8  # (N, 1)

#     u = mids - P0_exp             # (N, 6, 2)
#     alpha = (u * v[:, None, :]).sum(dim=2) / v_norm_sq    # (N, 6)

#     proj_loss = (F.relu(alpha - 1) ** 2 + F.relu(-alpha) ** 2).sum() / N
#     shrink_loss = ((P4 - P0) ** 2).sum(dim=1).mean()  # scalar
#     # center_loss = ((mids - center_exp).pow(2).sum(dim=2)).sum() / N
#     return lambda_proj * proj_loss
#     # return lambda_proj * proj_loss + lambda_center * center_loss


def bezier_shape_regularizer(control_points, bezier_degree=3, lambda_proj=1e-2):
    """
    Generalized shape regularizer for symmetric Bézier curve pairs.
    Args:
        control_points: (N, K, 2), where K = 2 * degree + 2
        bezier_degree: int, Bézier degree (>=2)
    Returns:
        Scalar shape loss
    """
    N, K, _ = control_points.shape
    # print(K, bezier_degree)
    assert K == 2 * bezier_degree + 2, "Control points shape mismatch with bezier_degree"

    # Endpoint: assume symmetric pair with shared endpoints at 0 and degree*2+1
    P0 = control_points[:, 0, :]                     # (N, 2)
    Pend = control_points[:, bezier_degree + 1, :]   # (N, 2)
    center = 0.5 * (P0 + Pend)                       # (N, 2)

    # Exclude endpoints: use all control points except 0 and degree+1
    idx = [i for i in range(K) if i != 0 and i != bezier_degree + 1]
    mids = control_points[:, idx, :]                 # (N, K-2, 2)

    # Project mids onto the line from P0 to Pend
    v = Pend - P0                                    # (N, 2)
    v_norm_sq = (v ** 2).sum(dim=1, keepdim=True) + 1e-8  # (N, 1)
    u = mids - P0[:, None, :]                        # (N, K-2, 2)
    alpha = (u * v[:, None, :]).sum(dim=2) / v_norm_sq  # (N, K-2)

    # === Projection loss ===
    proj_outside = F.relu(alpha - 1)**2 + F.relu(-alpha)**2  # (N, K-2)
    proj_loss = proj_outside.sum() / N

    total_loss = lambda_proj * proj_loss
    return total_loss
    # return lambda_proj * proj_loss


# def bezier_shape_regularizer(control_points, lambda_proj=1e-2, lambda_endpoint_push=0):
#     """
#     control_points: (N, 8, 2)
#     Returns: scalar loss
#     """
#     N = control_points.size(0)
#     P0 = control_points[:, 0, :]  # (N, 2)
#     P4 = control_points[:, 4, :]  # (N, 2)

#     mids = control_points[:, [1, 2, 3, 7, 6, 5], :]  # (N, 6, 2)
#     P0_exp = P0[:, None, :]       # (N, 1, 2)

#     v = P4 - P0                   # (N, 2)
#     v_norm = v.norm(dim=1, keepdim=True) + 1e-8     # (N, 1)
#     v_dir = v / v_norm                              # (N, 2)
#     v_norm_sq = v_norm ** 2                         # (N, 1)

#     u = mids - P0_exp             # (N, 6, 2)
#     alpha = (u * v_dir[:, None, :]).sum(dim=2)      # (N, 6)

#     # === proj loss: keep within [0, 1] ===
#     proj_loss = (F.relu(alpha - 1) ** 2 + F.relu(-alpha) ** 2).sum() / N

#     # === new loss: push P0/P4 outward if any control point is outside
#     # For alpha < 0 → encourage increasing (P0 - P4) length
#     # For alpha > 1 → encourage increasing (P4 - P0) length
#     push_P0 = F.relu(-alpha)     # (N, 6)
#     push_P4 = F.relu(alpha - 1)  # (N, 6)

#     # 以平均强度来计算两个方向上的 push 力度
#     push_P0_strength = push_P0.mean(dim=1, keepdim=True)  # (N, 1)
#     push_P4_strength = push_P4.mean(dim=1, keepdim=True)  # (N, 1)

#    # 虚拟目标点（不可导），用于构造 loss
#     P0_target = P0 - push_P0_strength * v_dir      # (N, 2)
#     P4_target = P4 + push_P4_strength * v_dir      # (N, 2)

#     # 端点推动 loss（总是非负）
#     push_loss = ((P0 - P0_target.detach()) ** 2).sum(dim=1) + \
#                 ((P4 - P4_target.detach()) ** 2).sum(dim=1)
#     endpoint_loss = push_loss.mean()

#     total_loss = lambda_proj * proj_loss + lambda_endpoint_push * endpoint_loss
    
#     return total_loss


def compute_sine_theta(s1, s2):
    # s1, s2: (2,2) tensor, representing a segment as [start, end]
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
    return sine_theta

def test_xing_loss_consistent():
    num_points = 10
    radius = 10.0
    angles = torch.linspace(0, math.pi/2, steps=num_points)
    x = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)  # shape: (10, 2)
    x_tensor = x.unsqueeze(0)  # shape: (1, 10, 2)
    
    loss = xing_loss(x_tensor)
    print("Loss for consistent turning curve (expected near 0):", loss.item())

def test_xing_loss_inconsistent():
    num_points = 10
    radius = 10.0
    angles = torch.linspace(0, math.pi/2, steps=num_points)
    x = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    x[5, :] = x[5, :] + torch.tensor([5.0, -5.0])
    x_tensor = x.unsqueeze(0)  # shape: (1, 10, 2)
    
    loss = xing_loss(x_tensor)
    print("Loss for inconsistent turning curve (expected > 0):", loss.item())

class sparse_coord_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', quantile_interval=200, nodiff_thres=0.05):
        print("pred shape and gt shape: ", pred.shape, gt.shape)
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0])**2).sum(0)
            # self.map = (np.abs(pred[0] - gt[0])).sum(0)
            self.reference_gt = copy.deepcopy(
                np.transpose(gt[0], (1, 2, 0)))
        elif format == ['[2D x c]']:
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError
        # OptionA: Zero too small errors to avoid the error too small deadloop
        self.map[self.map < nodiff_thres] = 0
        quantile_interval = np.linspace(0., 1., quantile_interval)
        quantized_interval = np.quantile(self.map, quantile_interval)
        # remove redundant
        quantized_interval = np.unique(quantized_interval)
        quantized_interval = sorted(quantized_interval[1:-1])
        self.map = np.digitize(self.map, quantized_interval, right=False)
        self.map = np.clip(self.map, 0, 255).astype(np.uint8)
        print("map shape is: ", self.map.shape)
        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = (self.map==idi).sum()
        self.idcnt.pop(min(self.idcnt.keys()))
        # remove smallest one to remove the correct region
    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            print("random")
            return [npr.uniform(0, 1)*h, npr.uniform(0, 1)*w]
        # target_id = max(self.idcnt, key=self.idcnt.get)
        target_id = max(self.idcnt, key=lambda k: k * self.idcnt[k])
        tolerance = 1
        mask = ((self.map >=target_id - tolerance) & (self.map <= target_id + tolerance + 2) &  (self.map > 0)).astype(np.uint8)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            ((mask)).astype(np.uint8), connectivity=4)

        # _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
        #     ((self.map == target_id)).astype(np.uint8), connectivity=4)
        # remove cid = 0, it is the invalid area
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize))+1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord-center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]
        # replace_sampling
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] == 0:
            self.idcnt.pop(target_id)
        self.map[component == target_cid] = 0
        return [coord_h, coord_w]
        # return [coord_w, coord_h]

    def get_current_region_mask(self, save_dir):
        if len(self.idcnt) == 0:
            return None

        target_id = max(self.idcnt, key=lambda k: k * self.idcnt[k])
        # _, component = cv2.connectedComponents((self.map == target_id).astype(np.uint8), connectivity=4)
        tolerance = 3
        mask = ((self.map >=target_id - tolerance) & (self.map <= target_id + tolerance + 2) & (self.map > 0)).astype(np.uint8)
        _, component = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        mask = (component > 0).astype(np.uint8) * 255  # [H, W]

        # mask = ((self.map == target_id).astype(np.uint8))
        num_labels, component = cv2.connectedComponents(mask, connectivity=4)
        binary_mask = (component > 0).astype(np.uint8) * 255  # For saving raw mask
        vis_color = label2rgb(component, bg_label=0, bg_color=(0, 0, 0))  # RGB [H, W, 3]
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.imsave(os.path.join(save_dir, "error_map.png"), self.map, cmap='viridis')
            plt.imsave(os.path.join(save_dir, f"region_mask_id_binary.png"), binary_mask, cmap='gray')
            plt.imsave(os.path.join(save_dir, f"region_mask_id_color.png"), vis_color)

        return binary_mask

if __name__ == '__main__':
    print("Test Consistent Turning:")
    test_xing_loss_consistent()
    print("\nTest Inconsistent Turning:")
    test_xing_loss_inconsistent()