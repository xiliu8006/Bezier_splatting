from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from torch.optim.lr_scheduler import LambdaLR
from gsplat.rasterize_sum import rasterize_gaussians_sum
from gsplat.rasterize import  rasterize_gaussians
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
from torchvision.ops import box_iou
from sh_utils import eval_sh
import torch.nn.functional as F
import random
import torch.distributions as dist
import time

def custom_lr_schedule(step):
    if step < 5000:
        return 1.0                # 初始大 lr
    elif step < 6000:
        return 0.5                # 再升
    elif step < 9000:
        return 0.2                # 再升
    # elif step < 11500:
    #     return 0.2                # 再升
    else:
        return 0.1              # 最后降

def time_cuda(func, name=""):
    torch.cuda.synchronize()
    start = time.time()
    result = func()
    torch.cuda.synchronize()
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"{name} took {elapsed_ms:.2f} ms")
    return result

def register_gradient_hook(tensor, name="tensor"):
    def hook(grad):
        print(f"[Grad Hook] {name} grad stats:")
        print(f"  shape: {grad.shape}")
        print(f"  mean:  {grad.mean().item():.6f}")
        print(f"  std:   {grad.std().item():.6f}")
        print(f"  max:   {grad.max().item():.6f}")
        print(f"  min:   {grad.min().item():.6f}")
        return grad  # important! don’t block gradient
    tensor.register_hook(hook)


# from pykeops.torch import LazyTensor
class FeatureAreaModulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_mask):
        ctx.save_for_backward(weight_mask)
        return input  # just pass through forward

    @staticmethod
    def backward(ctx, grad_output):
        weight_mask, = ctx.saved_tensors
        # print("shape is weight mask: ", weight_mask.shape, grad_output.shape)
        return grad_output * weight_mask, None  # apply weight to grad only

class PointGradientSmoother(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # input: [N, 2, H, 2]
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [N, 2, H, 2]
        N, B, H, D = grad_output.shape  # B=2, D=2 (x, y)

        # reshape for depthwise conv1d: merge batch + curve dim
        grad = grad_output.view(N * B, H, D).permute(0, 2, 1)  # [N*2, 2, H]

        kernel = torch.tensor([[0.25, 0.5, 0.25],
                               [0.25, 0.5, 0.25]], device=grad.device, dtype=grad.dtype).view(2, 1, 3)

        smoothed = F.conv1d(grad, kernel, padding=1, groups=2)  # depthwise on x/y
        smoothed = smoothed.permute(0, 2, 1).view(N, B, H, D)
        return smoothed


class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        # self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.ori_H, self.ori_W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.iter = 0
        self.device = kwargs["device"]
        self.mode = kwargs['mode']
        self.num_curves_init = 4096
        # self.num_curves = int(self.num_curves_init / 8)
        # self.num_curves = self.num_curves_init
        self.num_curves = kwargs["num_curves"]
        if self.mode == 'closed':
            self.num_beziers = 2 * 1
        else:
            self.num_beziers = 3
        self.opacity_mode = 0 # 0 is single opacity, 1 is multi opacity
        self.bezier_degree = 4
    
        self.curves_resolution = 40
        self.max_sh_degree = 1
        self.radius = 0.02
        if self.mode == "line":
            # self.num_samples = int(self.total_pixel / (self.num_curves * 20 * self.num_beziers))
            self.total_num_sample= self.num_samples
        elif self.mode == "unclosed":
            self.num_samples = 64
            self.total_num_sample= self.num_samples * self.num_beziers
            self.radius = 0.01
        elif self.mode == "closed":
            self.num_samples = kwargs["num_samples"]
            # self.num_samples = 32
            print("default num_samples: ", self.num_samples)
            self.total_num_sample = self.num_samples * self.num_beziers
        else:
            self.num_samples = 32
            # self.total_num_sample= self.num_samples * self.num_beziers
            self.total_num_sample= self.num_samples * self.num_beziers + self.curves_resolution**2
        
        self.rotation_activation = torch.sigmoid

        if self.mode == "line":
            self._control_points = self._initialize_control_points_line()
        elif self.mode == "closed":
            self._control_points = self._initialize_control_points()
        elif self.mode == "unclosed":
            self._control_points = self._initialize_control_points_line(8)
        else:
            self._control_points = self._initialize_control_points()
        self._features_dc = nn.Parameter(torch.rand(self.num_curves, 3))
        self._cholesky = nn.Parameter(torch.rand(self.num_curves, 3))

        self._scaling = nn.Parameter(torch.ones(self.num_curves, 1) * 2)
        self._rotation = nn.Parameter(torch.zeros(self.num_curves, 1))

        self._xyz = nn.Parameter(torch.zeros(self.num_curves, self.num_samples, 2))
        self._depth = nn.Parameter(torch.ones(self.num_curves, 1))
        if self.opacity_mode == 1:
            self._opacity = nn.Parameter(torch.ones(self.num_curves, 3))
        else:
            self._opacity = nn.Parameter(torch.ones(self.num_curves, 1))
        self._bernstein_cache = {}

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)


        l = [
            {'params': [self._control_points], 'lr': kwargs["lr"] * 0.02, "name": "control_points"},
            # {'params': [self._xyz], 'lr': kwargs["lr"] * 0, "name": "xyz"},
            {'params': [self._features_dc], 'lr': kwargs["lr"], "name": "features_dc"},
            {'params': [self._cholesky], 'lr': kwargs["lr"], "name": "cholesky"},
            {'params': [self._scaling], 'lr': kwargs["lr"], "name": "scaling"},
            # {'params': [self._rotation], 'lr': kwargs["lr"], "name": "rotation"},
            {'params': [self._opacity], 'lr': kwargs["lr"], "name": "opacity"},
            {'params': [self._depth], 'lr': kwargs["lr"], "name": "depth"}
        ]

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(l, lr=kwargs["lr"])
        else:
            self.optimizer = Adan(l, lr=kwargs["lr"])

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=custom_lr_schedule)
        self._update_bernstein_cache(self.bezier_degree + 1, self.num_samples, self.device)
        self._update_bernstein_cache(self.bezier_degree + 1, self.num_samples * 2, self.device)
        self._update_bernstein_cache(self.bezier_degree + 1, self.num_samples * 4, self.device)
        self._update_bernstein_cache(self.bezier_degree + 1, self.num_samples * 8, self.device)
        self._update_bernstein_cache(self.bezier_degree + 1, self.num_samples * 16, self.device)

        self.init_area_weight(self.device)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)
    
    def _update_bernstein_cache(self, n: int, num_samples: int, device: torch.device):
        key = (n, num_samples, str(device))
        if key in self._bernstein_cache:
            return

        t = torch.linspace(0.007, 0.993, num_samples, device=device)
        comb = torch.tensor([math.comb(n, i) for i in range(n + 1)],
                            dtype=torch.float32, device=device)
        comb_deriv = torch.tensor([math.comb(n - 1, i) for i in range(n)],
                                  dtype=torch.float32, device=device)
        t_pow = t[:, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)
        one_minus_t_pow = (1 - t[:, None]) ** torch.arange(n, -1, -1, dtype=torch.float32, device=device)

        t_pow_deriv = t[:, None] ** torch.arange(n, dtype=torch.float32, device=device)
        one_minus_t_pow_deriv = (1 - t[:, None]) ** torch.arange(n - 1, -1, -1, dtype=torch.float32, device=device)
        bernstein = comb * one_minus_t_pow * t_pow  # (num_samples, n + 1)
        bernstein_deriv = comb_deriv * one_minus_t_pow_deriv * t_pow_deriv  # (num_samples, n)

        self._bernstein_cache[key] = {
            'bernstein': bernstein,  # (num_samples, n + 1)
            'bernstein_deriv': bernstein_deriv  # (num_samples, n)
        }
    
    def _initialize_control_points(self):
        """
        Initialize control points for Bézier curves with an initial convex shape,
        distributed across the entire image.

        Returns:
        - control_points: A tensor of shape (num_curves, 12, 2).
        """
        num_segments = self.num_beziers  # Each curve has 3 Bézier segments
        num_points_per_curve = num_segments * (self.bezier_degree + 1)

        # Step 1: Generate random angles and radii for each curve
        angles = torch.linspace(0, 2 * torch.pi, num_points_per_curve, device=self.device)
        angles = angles.unsqueeze(0).expand(self.num_curves, -1)  # Shape: (num_curves, 13)

        radii = (torch.rand(self.num_curves, num_points_per_curve, device=self.device) * 0.5 + 0.5) * self.radius
        # Generate random curve centers within a normalized space [-1, 1]
        x_center = (torch.rand(self.num_curves, 1, 2, device=self.device) - 0.5) * 2  # Shape: (num_curves, 1, 2)
        # Convert polar coordinates to Cartesian relative to center
        x = x_center[:, :, 0] + radii * torch.cos(angles)  # Shape: (num_curves, num_points_per_curve, 1)
        y = x_center[:, :, 1] + radii * torch.sin(angles)  # Shape: (num_curves, num_points_per_curve, 1)

        # Combine x and y coordinates
        points = torch.stack([x, y], dim=-1)   # Shape: (num_curves, num_points_per_curve, 2)

        # Step 2: Add small perturbations to ensure diversity while maintaining convexity
        perturbation = torch.randn_like(points) * (self.radius * 0.05)
        points = points + perturbation

        # Step 3: Ensure the curve is closed (last point == first point)
        points[:, -1] = points[:, 0]

        # Step 4: Organize into control points (12 points per curve, 4 Bézier segments)
        control_points = points[:, :-1]  # Remove the repeated last point
        control_points = points
        # print("point shape  is : ",control_points.max(), control_points.min())

        return torch.nn.Parameter(control_points)
    
    def _initialize_control_points_line(self, order_beizer=2):
        """
        Initialize control points for the Bézier curves using vectorized operations.

        Returns:
        - control_points: A tensor of shape (num_curves, 4, 2).
        """
        # Step 1: Generate p0 (random values in range [0, 1])
        p0 = (torch.rand(self.num_curves, 1, 2, device=self.device) - 0.5) * 2

        # Step 2: Generate random offsets for p1, p2, p3 relative to the previous point
        # offsets = torch.randn(self.num_curves, order_beizer + 1, 2, device=self.device) * self.radius
        offsets = (torch.rand(self.num_curves, order_beizer + 1, 2, device=self.device) * 0.5 + 0.5) * self.radius


        # Step 3: Accumulate offsets to get relative positions
        relative_points = torch.cumsum(offsets, dim=1)

        # Step 4: Concatenate p0 and relative points to form the control points
        control_points = torch.cat([p0, p0 + relative_points], dim=1)
        print("control_points shape: ", control_points.shape)

        return torch.nn.Parameter(control_points)
    
    def _initialize_control_points_with_center(self, centers, radii=0.02):
        """
        Initialize control points for Bézier curves with an initial convex shape,
        distributed across the entire image.

        Returns:
        - control_points: A tensor of shape (num_curves, 12, 2).
        """
        x_centers = centers.clone()
        x_centers[:,:, 1] = (centers[:,:, 0] / self.H - 0.5) * 2
        x_centers[:,:, 0] = (centers[:,:, 1] / self.W - 0.5) * 2
        # print(centers[:,:, 0].)
        num_segments = self.num_beziers  # Each curve has 3 Bézier segments
        if self.mode == 'unclosed':
            num_points_per_curve = num_segments * 3 + 1  
            num_offsets = num_points_per_curve - 1      
            p0 = x_centers  # (N, 1, 2)
            num_left = num_offsets // 2
            num_right = num_offsets - num_left
            offsets_left = (torch.rand(x_centers.shape[0], num_left, 2, device=self.device) * 0.5 + 0.5) * 0.005
            offsets_right = (torch.rand(x_centers.shape[0], num_right, 2, device=self.device) * 0.5 + 0.5) * 0.005
            relative_left = -torch.cumsum(offsets_left, dim=1)
            relative_right = torch.cumsum(offsets_right, dim=1)
            points_left = p0 + relative_left.flip(dims=[1])  # (N, num_left, 2)
            points_right = p0 + relative_right               # (N, num_right, 2)
            print("shape: ", points_left.shape, p0.shape, points_right.shape)
            control_points = torch.cat([points_left, p0, points_right], dim=1)  # (N, num_points, 2)
            return torch.nn.Parameter(control_points)
        else:
            num_points_per_curve = num_segments * (self.bezier_degree + 1)  # Total: 13 points (last point repeats the first)
            num_curves = centers.shape[0]

            # Step 1: Generate random angles and radii for each curve
            angles = torch.linspace(0, 2 * torch.pi, num_points_per_curve, device=self.device)
            angles = angles.unsqueeze(0).expand(num_curves, -1)  # Shape: (num_curves, 13)
            x = x_centers[:, :, 0] + radii * torch.cos(angles)  # Shape: (num_curves, num_points_per_curve, 1)
            y = x_centers[:, :, 1] + radii * torch.sin(angles)  # Shape: (num_curves, num_points_per_curve, 1)
            points = torch.stack([x, y], dim=-1)   # Shape: (num_curves, num_points_per_curve, 2)
            points[:, -1] = points[:, 0]
            control_points = points[:, :-1]  # Remove the repeated last point
            control_points = points
            return torch.nn.Parameter(control_points)
    
    def compute_valid_aabb(self, points):
        """
        Compute the valid axis-aligned bounding box (AABB) for points within [-1, 1].
        Points outside this range are ignored when computing the min/max bounds.
        """
        # Mask indicating which points are valid (within [-1, 1])
        mask = (points >= -1) & (points <= 1)  # shape: (N, M, 2)
        
        # Constants for replacing invalid points
        inf_val  = torch.tensor(float('inf'), device=points.device, dtype=points.dtype)
        ninf_val = torch.tensor(float('-inf'), device=points.device, dtype=points.dtype)
        
        # For min(): replace invalid points with +inf so they are ignored
        points_for_min = torch.where(mask, points, inf_val)
        bbox_min = points_for_min.min(dim=1).values  # shape: (N, 2)
        
        # For max(): replace invalid points with -inf so they are ignored
        points_for_max = torch.where(mask, points, ninf_val)
        bbox_max = points_for_max.max(dim=1).values  # shape: (N, 2)
        
        # If a point set has no valid points in some dimension, its bbox_min = inf
        # and bbox_max = -inf. Assign a default value (-1) in such cases.
        bbox_min = torch.where(torch.isinf(bbox_min), torch.full_like(bbox_min, -1.0), bbox_min)
        bbox_max = torch.where(torch.isinf(bbox_max), torch.full_like(bbox_max, -1.0), bbox_max)
        
        return bbox_min.unsqueeze(1), bbox_max.unsqueeze(1)

    # @property
    def get_scaling(self, factor=1):
        if self.mode == 'closed':
            return self.get_scaling_closed(factor)
        else:
            return self.get_scaling_open()

    def get_scaling_closed(self, factor):
        xyz = torch.cat([self.xyz, self.xyz_area], dim=1).detach()
        N = xyz.shape[1] 
        diffs = torch.abs(xyz[:, :, 1:, :] - xyz[:, :, :-1, :])
        scale = torch.tensor([self.W * factor, self.H * factor], device=diffs.device).view(1, 1, 1, 2)
        diffs = diffs * scale
           
        sigma = torch.norm(diffs, dim=-1)
        sigma_last  = sigma[:, :, -2:-1].clone()  # shape: [N, 1] 
        sigma_x = torch.cat([sigma, sigma_last], dim=-1) / (3.0 / torch.sqrt(torch.tensor(factor, dtype=torch.float32)))
        scale = torch.tensor([0.4, 0.9, 1.0], device=sigma_x.device).view(1, 1, 3)
        sigma_x[:, :, :3] *= scale
        sigma_x[:, :, -3:] *= scale.flip(dims=[2])
        sigma_x[:, :2, :].clamp_(min=0.3)
        # sigma_x.clamp_(min=0.5)

        index_order = torch.arange(2, N, device=xyz.device)
        index_order = torch.cat([torch.tensor([0], device=xyz.device), index_order, torch.tensor([1], device=xyz.device)])
        xyz_reordered = xyz[:, index_order, :, :].clone()
        
        diffs_y = torch.abs(xyz_reordered[:, 1:, :, :] - xyz_reordered[:, :-1, :, :])
        diffs_y[:, :, :, 0] *= self.W * factor
        diffs_y[:, :, :, 1] *= self.H * factor
        sigma_ = torch.norm(diffs_y, dim=-1)
        sigma_first = sigma_[:, :1, :].clone() 
        sigma_y = torch.cat([sigma_first, sigma_], dim=1) / (3.0 / torch.sqrt(torch.tensor(factor, dtype=torch.float32))) # 归一化

        sigma_y[:, :2, :].clamp_(max=1.0, min=0.75)

        threshold = 0.1
        ratio = 3.0

        sx = sigma_x.clone()
        sy = sigma_y.clone()
        # mask where either value is extremely small
        mask = (sy < threshold)
        # only apply ratio clamp on positions starting from index 2
        mx = mask[:, 2:, :]
        my = mask[:, 2:, :]
        # sigma_x clamp only where mask = True
        sigma_x[:, 2:, :] = torch.where(
            mx, 
            torch.min(sx[:, 2:, :], sy[:, 2:, :] * ratio),
            sx[:, 2:, :]
        )
        # sigma_y clamp only where mask = True
        sigma_y[:, 2:, :] = torch.where(
            my,
            torch.min(sy[:, 2:, :], sx[:, 2:, :] * ratio),
            sy[:, 2:, :]
        )
        scaling = torch.cat([sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1)], dim=-1).contiguous()
        # print("scaling max and min: ", scaling.max(), scaling.min())
        return scaling.view(-1, 2).detach()
    
    def get_scaling_open(self):
        xyz = self.xyz.view(self._control_points.shape[0], self.total_num_sample, 2).detach()
        diffs = torch.abs(xyz[:, 1:, :] - xyz[:, :-1, :])
        diffs[:, :, 0] *= self.W
        diffs[:, :, 1] *= self.H
        
        sigma_ratio = 2
        sigma = torch.norm(diffs, dim=2)
        sigma_last  = sigma[:, -1:].clone()  # shape: [N, 1] 
        sigma_x = (torch.cat([sigma, sigma_last], dim=1)) / sigma_ratio + 0.5
        sigma_y = torch.abs(self._scaling.repeat_interleave(self.total_num_sample, dim=1) + 0.5)
        # print("max scale: ", sigma_x.max(), sigma_x.mean(), sigma_y.max(), sigma_y.mean())
        scaling = torch.cat([sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1)], dim=-1)
        return scaling.view(-1, 2)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation.repeat_interleave(self.num_samples, dim=0))*2*math.pi

    @property
    def get_xyz(self):
        # return torch.tanh(self._xyz)
        if self.mode == "line":
            # xyz, normals = self.sample_bezier_curves(self._control_points, self.num_samples)
            xyz, normals, tangents = self.sample_bezier_curves(self._control_points, self.num_samples)
        elif self.mode == "unclosed":
            xyz, normals, tangents = self.sample_bezier_curves_unclose(self._control_points, self.total_num_sample)
        else:
            xyz, normals, tangents = self.sample_boundary(self._control_points, self.num_samples)
        # return nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        return xyz.reshape(-1, 2), normals, tangents
    
    def get_xyz_and_depth(self, factor=1, denser_sample=False):
        # print("self mode is: ", self.mode)
        if self.mode == "line":
            xyz, normals, tangents = self.sample_bezier_curves(self._control_points, self.num_samples * factor)
        elif self.mode == "unclosed":
            xyz= self.sample_bezier_curves_unclose(self._control_points, self.total_num_sample * factor)
            return xyz.reshape(-1, 2), torch.zeros(1)
        else:
            if denser_sample:
                sampled_points, area_points = self.sample_bezier_area(self._control_points, resolution=self.curves_resolution * factor, factor=factor)
            else:
                sampled_points, area_points = self.sample_bezier_area(self._control_points, resolution=self.curves_resolution)

            # print("self.xyz.shape:",  sampled_points.shape)
            return sampled_points, area_points
            # smoothed_points = PointGradientSmoother.apply(sampled_points)  # points: [N, M, 2]
            # return smoothed_points, area_points
        return xyz
    
    @property
    def get_xyz(self):
        xyz, normals, tangents = self.sample_bezier_curves(self._control_points, self.num_samples)
        return xyz.view(-1, 2)
    
    @property
    def get_samples(self):
        # return torch.tanh(self._xyz)
        xyz, normals, tangents = self.sample_bezier_curves_uniform(self._control_points, self.num_samples)
        # return nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        return xyz, normals
    
    @property
    def get_features(self):
        if self.mode == 'closed':
            return self.get_features_closed()
        else:
            return self.get_features_open()
    
    def get_area_weight(self, shape, device, alpha=4.0):
        """
        shape: (B, B_area, H, D)
        return: (B, B_area, H, D)
        """
        _, B_area, H, D = shape

        yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)      # [1, 1, H, 1]
        xx = torch.linspace(-1, 1, B_area, device=device).view(1, B_area, 1, 1)  # [1, B_area, 1, 1]

        dist = torch.abs(yy) + torch.abs(xx)  
        weight = 1.0 - torch.exp(-alpha * dist) 

        return weight.repeat(shape[0], 1, 1, D)  # [B, B_area, H, D]
    
    def init_area_weight(self, device, alpha=4.0):
        H = int(self.total_num_sample / 2)
        B_area =  self.curves_resolution
        D = self._features_dc.shape[-1]
        yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)      # [1, 1, H, 1]
        xx = torch.linspace(-1, 1, B_area, device=device).view(1, B_area, 1, 1)  # [1, B_area, 1, 1]
        dist = torch.abs(yy) + torch.abs(xx)  # [1, B_area, H, 1]，中心为0，边缘为最大
        weight = 1.0 - torch.exp(-alpha * dist)  # 
        self.area_weight = weight.repeat(self.num_curves, 1, 1, D)  # [B, B_area, H, D]

    def get_features_closed(self):
        features_dc = self._features_dc.unsqueeze(1).unsqueeze(1).repeat(1, self.xyz.shape[1], self.xyz.shape[2], 1)
        features_dc_area = self._features_dc.unsqueeze(1).unsqueeze(1).repeat(1, self.xyz_area.shape[1], self.xyz.shape[2], 1)
        area_weight = self.get_area_weight(features_dc_area.shape, features_dc_area.device)
        features_dc_area = FeatureAreaModulator.apply(features_dc_area, area_weight)
        features = torch.cat([features_dc, features_dc_area], dim=1)
        _features_dc_expanded = torch.sigmoid(features)
        return _features_dc_expanded
    
    def get_features_open(self):
        _features_dc_expanded = torch.clamp(self._features_dc.unsqueeze(1).expand(-1, self.total_num_sample, -1),min=0.0, max=1.0)
        return _features_dc_expanded.view(-1, 3)
    
    @property
    def get_depth(self):
        if self.mode == "closed":
            depth = self._depth.unsqueeze(2).repeat(1, self.xyz.shape[1] + self.xyz_area.shape[1], self.xyz.shape[2])
            depth_clone = depth.clone().detach()
            depth_clone[:, :2, :] -= 1e-6
            return torch.sigmoid(depth_clone)
        else:
            return torch.sigmoid(self._depth.repeat_interleave(self.total_num_sample, dim=0))

    def compute_rotations(self, points):
        """
        Compute the rotation angle at each sampled point along each curve.

        Args:
            points: Tensor of shape [N, H, 2]
                - N: number of curves
                - H: number of sampled points per curve
                - Each point is a 2D coordinate.

        Returns:
            rotations: Tensor of shape [N, H]
                - Rotation angle (in radians) at each sampled point.
                - For the last sampled point, the angle is copied from the 
                second-to-last point.
        """
        if self.mode == 'closed':
            xyz = torch.cat([self.xyz, self.xyz_area], dim=1).detach()
        else:
            xyz = points.detach().view(self._control_points.shape[0], self.num_beziers, -1, 2)
        diffs = xyz[:, :, 2:, :] - xyz[:, :, :-2, :]
        diffs[:, :, :, 0] *= self.W
        diffs[:, :, :, 1] *= self.H
        
        theta = torch.atan2(diffs[..., 1], diffs[..., 0])  # shape: [N, H-1]
        theta_first = theta[..., :1].clone()  # shape: [N, 1] 
        theta_last = theta[..., -1:].clone()
        rotations = torch.cat([theta_first, theta, theta_last], dim=-1)
        return -rotations

    def bezier_interpolate(self, input_tensor, num_samples):
        """
        Perform cubic Bézier interpolation for each row of input data.

        Args:
            input_tensor (Tensor): Shape [N, 4].  
                Each row contains 4 control points (supporting 1D scalar control points).

            num_samples (int):  
                Number of sampling points to generate along each Bézier curve.

        Returns:
            Tensor: Shape [N, num_samples].  
                Each row contains uniformly sampled values from the Bézier curve defined by
                the corresponding control points.
        """
        # Generate parameter t: uniformly sampled values in [0, 1] with num_samples points
        t = torch.linspace(0, 1, num_samples, device=input_tensor.device).unsqueeze(1)  # shape: [num_samples, 1]
        one_minus_t = 1 - t

        B0 = one_minus_t ** 3               # (1-t)^3
        B1 = 3 * t * (one_minus_t ** 2)       # 3t(1-t)^2
        B2 = 3 * (t ** 2) * one_minus_t       # 3t^2(1-t)
        B3 = t ** 3                         # t^3

        weights = torch.cat([B0, B1, B2, B3], dim=1)
        
        # input_tensor: [N, 4], weights.t(): [4, num_samples]
        output = torch.matmul(input_tensor, weights.t())
        return output

    @property
    def get_opacity(self):
        if self.mode == "closed":
            if self.opacity_mode == 1:
                N = self._opacity.shape[0]  
                L = self.xyz_area.shape[1] 
                M = self.xyz.shape[2] 

                opacities_first = self._opacity[:, :1]  # (N, 1)
                opacities_middle = self._opacity[:, 1:2]  # (N, 1)
                opacities_last = self._opacity[:, 2:]  # (N, 1)

                weights_first = torch.linspace(0, 1, steps=L // 2, device=self._opacity.device).view(1, -1)  # (1, L//2)
                weights_second = torch.linspace(0, 1, steps=L - L // 2, device=self._opacity.device).view(1, -1)  # (1, L-L//2)

                opacities_area_first_half = (1 - weights_first) * opacities_first + weights_first * opacities_middle  # (N, L//2)

                opacities_area_second_half = (1 - weights_second) * opacities_middle + weights_second * opacities_last  # (N, L - L//2)

                opacities_area = torch.cat([opacities_area_first_half, opacities_area_second_half], dim=1)

                opacities_area = opacities_area.unsqueeze(-1).repeat(1, 1, M)
                opacity = torch.cat([
                    opacities_first.unsqueeze(1).repeat(1, 1, M),
                    opacities_last.unsqueeze(1).repeat(1, 1, M),
                    opacities_area
                ], dim=1)  # (N, L+2, M)
                return self.opacity_activation(opacity.contiguous().view(-1, 1))
            else:
                opacities =  self._opacity.unsqueeze(1).repeat(1, self.xyz.shape[1] + self.xyz_area.shape[1], self.xyz.shape[2])
                return  self.opacity_activation(opacities.contiguous().view(-1, 1))
        else:
            N, cols = self._opacity.shape
            base_rep = self.total_num_sample // 3
            remainder = self.num_samples % 3
            parts = []
            for i in range(3):
                rep = base_rep + (remainder if i == 3 else 0)
                part = self._opacity[:, i].unsqueeze(1).repeat(1, rep)
                parts.append(part)
                out = torch.cat(parts, dim=1)
            return self.opacity_activation(out.contiguous().view(-1, 1))


    @property
    def get_cholesky_elements(self):
        _cholesky_expanded = self._cholesky.unsqueeze(1).repeat(1, self.num_samples, 1)
        return _cholesky_expanded.view(-1, 3) + self.cholesky_bound

    @property
    def get_beizer_curves(self):
        return self._control_points
    
    def sample_bezier_curves_uniform(self, bezier_curves: torch.Tensor, num_samples: int):
        """
        Sample Bézier curves and return sampled points, normals, and tangents.

        Args:
            bezier_curves: Tensor (num_curves, num_control_points, 2)
            num_samples: Number of points to sample per curve

        Returns:
            sampled_points: (num_curves, num_samples, 2)
            normals: (num_curves, num_samples, 2)
            tangents: (num_curves, num_samples, 2)
        """
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("Control points must be 2D coordinates.")

        device = bezier_curves.device
        n = num_control_points - 1  # Bézier degree

        # self._update_bernstein_cache(n, num_samples, device)
        key = (n, num_samples, str(device))
        cache = self._bernstein_cache[key]

        bernstein = cache['bernstein'][None, :, :]  # (1, num_samples, n+1)
        bernstein_deriv = cache['bernstein_deriv'][None, :, :]  # (1, num_samples, n)

        # Sampled points: (num_curves, num_samples, 2)
        sampled_points = torch.sum(bernstein[..., None] * bezier_curves[:, None, :, :], dim=2)
        return sampled_points

    def compute_aabb(self, points_tensor):
        """
        Args:
            points_tensor (torch.Tensor): (N, H, 2)。

        Returns:
            torch.Tensor: AABB,  (N, 4) (x1, y1, x2, y2)。
        """
        if points_tensor.ndim != 3 or points_tensor.size(-1) != 2:
            raise ValueError("points_tensor must have shape (N, H, 2).")

        min_coords, _ = points_tensor.min(dim=1)  # Shape: (N, 2)
        max_coords, _ = points_tensor.max(dim=1)  # Shape: (N, 2)

        aabb = torch.cat([min_coords, max_coords], dim=1)  # Shape: (N, 4)
        return aabb

    def compute_aabb_area(self, boxes):
        """
        Args:
            boxes (torch.Tensor):  (N, 4), (x1, y1, x2, y2)。

        Returns:
            torch.Tensor:(N,)。
        """
        if boxes.ndim != 2 or boxes.size(1) != 4:
            raise ValueError("boxes must have shape (N, 4).")
        

        widths = boxes[:, 2] - boxes[:, 0]  # x2 - x1
        heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1

        areas = widths * heights
        return areas


    def replace_points(self, points_tensor, target=(0, 0)):
        """
        Replace points in `points_tensor` that match the given target value with
        other non-target values from the same group.

        Args:
            points_tensor (torch.Tensor): 
                Input point set of shape (N, H, 2).

            target (tuple):
                The value to be replaced. Default is (-2, -2).

        Returns:
            torch.Tensor:
                The updated point set after replacing all target values.
        """

        if points_tensor.ndim != 3 or points_tensor.size(-1) != 2:
            raise ValueError("points_tensor must have shape (N, H, 2).")

        target_tensor = torch.tensor(target, device=points_tensor.device, dtype=points_tensor.dtype) 
        mask = (points_tensor < target_tensor).all(dim=-1)

        for n in range(points_tensor.size(0)):
            valid_points = points_tensor[n, ~mask[n]]
            if valid_points.size(0) == 0:
                return points_tensor
            replacement_value = valid_points[0]
            points_tensor[n, mask[n]] = replacement_value

        return points_tensor

    
    def compute_outside_area(self, boxes):
        H, W = self.H, self.W
        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        x_max = boxes[:, 2]
        y_max = boxes[:, 3]

        widths = x_max - x_min
        heights = y_max - y_min
        total_area = widths * heights

        inter_x_min = torch.clamp(x_min, min=0, max=W)
        inter_y_min = torch.clamp(y_min, min=0, max=H)
        inter_x_max = torch.clamp(x_max, min=0, max=W)
        inter_y_max = torch.clamp(y_max, min=0, max=H)

        inter_width = (inter_x_max - inter_x_min).clamp(min=0)
        inter_height = (inter_y_max - inter_y_min).clamp(min=0)
        inter_area = inter_width * inter_height
        outside_area = total_area - inter_area

        return outside_area
    
    def compute_pairwise_overlap(self, curve_1, curve_2, threshold=10):
        
        distances = torch.cdist(curve_1, curve_2, p=2)
        min_distances, _ = torch.min(distances, dim=1)
        min_distances, _ = torch.min(distances, dim=1)
        match_mask = min_distances < threshold
        match_count = int(torch.sum(match_mask).item())
        matched_distances = min_distances[match_mask]
        return match_count, matched_distances

    def remove_curves_mask(self):
        if self.mode == "closed":
            return self.remove_curves_mask_area()
        elif self.mode == "unclosed":
            return self.remove_curves_mask_line()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def remove_curves_mask_line(self, top_k=0.01, iou_threshold=0.1, color_threshold=0.05, remove_num=None, imagesize=None):
        color_threshold_input = color_threshold
        if self.iter < 7000:
            area_threshold = 5000
            color_threshold = color_threshold_input
        else:
            # color_threshold = color_threshold_input / 2
            area_threshold = 500
        xyz = self.xyz.view(-1,self.total_num_sample,2)
        num_curves = self._control_points.shape[0]
        xys = self.xys.view(num_curves, -1, 2).detach()
        boxes = self.compute_aabb(xys)
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        areas = widths * heights
        outside_area = self.compute_outside_area(boxes)
        ratio = outside_area / areas
        mask_outside = ratio > 0.6
        inter_left = torch.max(boxes[:, None, 0], boxes[None, :, 0])  # max(x1_1, x1_2)
        inter_top = torch.max(boxes[:, None, 1], boxes[None, :, 1])   # max(y1_1, y1_2)
        inter_right = torch.min(boxes[:, None, 2], boxes[None, :, 2]) # min(x2_1, x2_2)
        inter_bottom = torch.min(boxes[:, None, 3], boxes[None, :, 3])# min(y2_1, y2_2)

        inter_width = (inter_right - inter_left).clamp(min=0)
        inter_height = (inter_bottom - inter_top).clamp(min=0)
        inter_area = inter_width * inter_height

        ratio_matrix = inter_area / areas.unsqueeze(1)
        ratio_matrix.fill_diagonal_(0)
        color = self._features_dc.clone() * self.opacity_activation(self._opacity)
        color_diff = torch.norm(color.unsqueeze(1) - color.unsqueeze(0), dim=-1)
        
        keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)
        iou_mask = (ratio_matrix > iou_threshold)
        color_mask = (color_diff < color_threshold)
        suppress_matrix = iou_mask & color_mask
        suppress_matrix.fill_diagonal_(0)
        keep[mask_outside] = False

        xys = self.xys.clone().detach().view(self._control_points.shape[0], -1, 2)
        line_areas = torch.norm(xys[:, 1:, :] - xys[:, :-1, :], dim=-1).sum(-1)
        line_areas = line_areas * torch.abs(self._scaling.detach()).squeeze(-1)
        opacities = torch.sigmoid(self._opacity).squeeze(-1)
        opacities_threshold = 0.6
        if self.iter > 10000:
            areas_mask = line_areas < area_threshold
            opacities_mask = opacities.sum(-1) < opacities_threshold / 2
            print("line areas max: ", line_areas.max(), line_areas.min(),areas_mask.shape, opacities_mask.shape)
            keep[areas_mask & opacities_mask] = False
        else:
            opacities_mask = opacities < opacities_threshold
            keep[opacities.sum(-1) < opacities_threshold] = False

        for idx in range(len(areas)):
            if not keep[idx]:
                continue 
            if suppress_matrix[idx][idx + 1:].sum() > 0:
                slice_part = suppress_matrix[idx, idx+1:]
                relative_idx = (slice_part > 0).nonzero(as_tuple=False)
                original_idx = relative_idx + (idx + 1)
                total_match_count = 0.0
                for qualified_idx in original_idx:
                    match_count, matched_distances = self.compute_pairwise_overlap(xys[idx], xys[qualified_idx])
                    total_match_count += match_count
                # print("see the avg: ", (total_match_count / float(self.num_samples)))
                if self.iter > 10000:
                    if (total_match_count / float(self.num_samples) > 0.6) and line_areas[idx] < area_threshold:
                        keep[idx] = False
                else:
                    if (total_match_count / float(self.num_samples) > 0.6):
                        keep[idx] = False
            if (widths[idx]<4) & (heights[idx]<4) & (widths[idx] * heights[idx] < 12):
                keep[idx] = False
        return keep

    def remove_curves_mask_area(self, top_k=0.01, iou_threshold=0.1, color_threshold=0.05, remove_num=None, imagesize=None):
        color_threshold_input = color_threshold
        if self.iter < 6000:
            area_threshold = 50000
            color_threshold = color_threshold_input
        elif self.iter <8000:
            area_threshold = 20000
        else:
            area_threshold = 2000
        xyz = self.xyz.view(-1,self.total_num_sample,2)
        diffs = xyz[:, 1:, :] - xyz[:, :-1, :]
        distance = torch.norm(diffs, dim=2).sum(dim=1)
        if remove_num is not None:
            values, indices = torch.topk(self._grad / torch.min(torch.abs(self._cholesky[:, :1]), torch.abs(self._cholesky[:, 2:])), \
                                        remove_num, dim=0, largest=True)

        num_curves = self._control_points.shape[0]
        xys = self.xys.view(num_curves, -1, 2).detach()
        boxes = self.compute_aabb(xys)
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        areas = widths * heights
        outside_area = self.compute_outside_area(boxes)
        ratio = outside_area / areas
        mask_outside = ratio > 0.6
        inter_left = torch.max(boxes[:, None, 0], boxes[None, :, 0])  # max(x1_1, x1_2)
        inter_top = torch.max(boxes[:, None, 1], boxes[None, :, 1])   # max(y1_1, y1_2)
        inter_right = torch.min(boxes[:, None, 2], boxes[None, :, 2]) # min(x2_1, x2_2)
        inter_bottom = torch.min(boxes[:, None, 3], boxes[None, :, 3])# min(y2_1, y2_2)

        inter_width = (inter_right - inter_left).clamp(min=0)
        inter_height = (inter_bottom - inter_top).clamp(min=0)
        inter_area = inter_width * inter_height
        ratio_matrix = inter_area / areas.unsqueeze(1) 
        print("ratio_matrix.max: ", ratio_matrix.min(), ratio_matrix.max())
        ratio_matrix.fill_diagonal_(0)
        color = torch.sigmoid(self._features_dc.clone()) * self.opacity_activation(self._opacity)
        # color = self._features_dc.clone() * self.opacity_activation(self._opacity)
        print("color: ", color.shape, self._features_dc.shape, self._opacity.shape)
        # color_diff = torch.norm(self._features_dc.unsqueeze(1) - self._features_dc.unsqueeze(0), dim=-1)
        color_diff = torch.norm(color.unsqueeze(1) - color.unsqueeze(0), dim=-1)
        keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)
        iou_mask = (ratio_matrix > iou_threshold)
        color_mask = (color_diff < color_threshold)
        suppress_matrix = iou_mask & color_mask
        suppress_matrix.fill_diagonal_(0)
        remove_by_overlap = 0

        keep[mask_outside] = False
        locked_indices = set()
        for idx in range(len(areas)):
            if not keep[idx]:
                continue
            if idx in locked_indices:
                continue
            if suppress_matrix[idx][idx + 1:].sum() > 0:
                slice_part = suppress_matrix[idx, idx+1:]
                relative_idx = (slice_part > 0).nonzero(as_tuple=False)
                original_idx = relative_idx + (idx + 1)
                # total_match_count = 0
                iou_total = 0
                weight_iou_total = 0
                for qualified_idx in original_idx:
                    iou = ratio_matrix[idx, qualified_idx]
                    iou_color_diff = color_diff[idx, qualified_idx]
                    weight_iou_total += iou / (torch.sigmoid(iou_color_diff * 100) + 1e-2)
                    iou_total += iou
                if weight_iou_total > 0.5 and areas[idx] < area_threshold:
                    keep[idx] = False
                    remove_by_overlap += 1

                    for qualified_idx in original_idx:
                        locked_indices.add(qualified_idx)

            if remove_num is None:
                if (widths[idx]<5) & (heights[idx]<5) & (widths[idx] * heights[idx] < 16):
                    keep[idx] = False
        print("remove by overlap: ", remove_by_overlap)
        # opacities = torch.sigmoid(self._opacity).squeeze(-1)
        opacities = torch.sigmoid(self._opacity)
        if self.opacity_mode == 1:
            opacities_threshold = 0.1
            opacities_threshold_final = 0.05
            # opacities_threshold = 0.3
            # opacities_threshold_final = 0.6
        else:
            opacities_threshold = 0.2
            opacities_threshold_final = 0.2
        if self.iter > 7000:
            areas_mask = areas < area_threshold
            if self.opacity_mode == 1:
                opacities_mask_2 = opacities[:, 1] < opacities_threshold_final
                # keep[areas_mask & opacities_mask_2] = False
                opacities_threshold_final = 0.3
            opacities_mask = opacities.sum(-1) < opacities_threshold_final
            keep[areas_mask & opacities_mask] = False
            print("mask 1 and mask 2 : " , opacities_mask.sum(), opacities_threshold_final, opacities.shape)
        else:
            if self.opacity_mode == 1:
                opacities_mask_2 = opacities[:, 1] < opacities_threshold
                # keep[opacities_mask_2] = False
                opacities_threshold_final =0.6
            opacities_mask = opacities.sum(-1) < opacities_threshold
            keep[opacities_mask] = False
            print("mask 1 and mask 2 : " , opacities_mask.sum(), opacities_threshold_final, opacities.shape)

        return keep
  
    def compute_rotated_bbox_vertices(self, cx, cy, width, height, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        dx = width / 2
        dy = height / 2
        corners = np.array([
            [-dx, -dy],
            [ dx, -dy],
            [ dx,  dy],
            [-dx,  dy]
        ])

        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = np.dot(corners, rotation_matrix.T)

        translated_corners = rotated_corners + np.array([cx, cy])
        return translated_corners.tolist()

    def sample_bezier_curves_unclose(self, control_points, num_samples, fine_samples=1000):
        """
        Sample boundary points and normals from a set of closed Bézier curves.

        Parameters:
        - control_points: Tensor of shape (num_curves, 12, 2)
        - num_samples: Total number of samples per curve
        - fine_samples: Unused for now

        Returns:
        - sampled_points: (num_curves, 4 * num_samples, 2)
        """
        num_curves, total_control_points, _ = control_points.shape
        assert total_control_points == self.num_beziers * 3 + 1, (
            f"Expected {self.num_beziers * 3 + 1} control points, got {total_control_points}"
        )
        device = control_points.device
        samples_per_segment = int(num_samples / self.num_beziers)

        # Vectorized generation of control point indices for each Bézier segment
        base = torch.arange(self.num_beziers, device=device).unsqueeze(1) * 3  # (num_beziers, 1)
        offsets = torch.arange(4, device=device).unsqueeze(0)  # (1, 4)
        indices = (base + offsets) % total_control_points  # (num_beziers, 4)

        # Expand for all curves
        indices = indices.unsqueeze(0).expand(num_curves, -1, -1)  # (num_curves, num_beziers, 4)

        # Gather control points for each segment
        control_points_exp = control_points.unsqueeze(1).expand(-1, self.num_beziers, -1, -1)  # (num_curves, num_beziers, total_cp, 2)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, 2)  # (num_curves, num_beziers, 4, 2)

        segment_control_points = time_cuda(
            lambda: torch.gather(control_points_exp, 2, indices_exp),
            "extract_segment_control_points"
        )  # Shape: (num_curves, num_beziers, 4, 2)

        # Merge all segments into a flat batch
        merged_control_points = segment_control_points.reshape(-1, 4, 2)  # (num_curves * num_beziers, 4, 2)

        # Sample each Bézier segment
        sampled_points= time_cuda(
            lambda: self.sample_bezier_curves_uniform(
                merged_control_points,
                samples_per_segment,
            ),
            "sample_bezier_curves_uniform"
        )  # (num_curves * num_beziers, samples_per_segment, 2)

        # Reshape back to (num_curves, num_samples, 2)
        # print("sample points shape: ", sampled_points.shape, num_samples, samples_per_segment)
        # sampled_points = sampled_points.view(num_curves, num_samples, 2)
        # print("sample points shape: ", sampled_points.shape)
        return sampled_points


    def sample_bezier_curves(self, bezier_curves, num_samples, fine_samples=1000):
        """
        Generate uniformly spaced sample points and normals along Bézier curves
        using arc-length parameterization (revised version).

        Args:
            bezier_curves (torch.Tensor):
                Control points of the Bézier curves, shaped
                (num_curves, num_control_points, 2).

            num_samples (int):
                Number of target sample points to generate along each curve.

            fine_samples (int):
                Number of fine-grained samples used to approximate arc length.

        Returns:
            sampled_points (torch.Tensor):
                Uniformly distributed sample points along each curve,
                shaped (num_curves, num_samples, 2).

            normals (torch.Tensor):
                Corresponding normal vectors at each sampled point,
                shaped (num_curves, num_samples, 2).
        """
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("控制点必须是 2D 坐标。")

        device = bezier_curves.device
        n = num_control_points - 1

        t_values_fine = torch.linspace(0, 1, fine_samples, device=device)
        comb = torch.tensor([math.comb(n, i) for i in range(n + 1)], dtype=torch.float32, device=device)
        t_powers = t_values_fine[:, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)  # t^i
        one_minus_t_powers = (1 - t_values_fine[:, None]) ** torch.arange(n, -1, -1, dtype=torch.float32, device=device)  # (1-t)^(n-i)
        bernstein = comb * one_minus_t_powers * t_powers  # Bernstein (fine_samples, n+1)
        bernstein = bernstein.unsqueeze(0)  # (1, fine_samples, n+1)
        fine_points = torch.sum(bernstein[..., None] * bezier_curves[:, None, :, :], dim=2)  # (num_curves, fine_samples, 2)

        # Step 2
        deltas = torch.norm(fine_points[:, 1:, :] - fine_points[:, :-1, :], dim=-1)
        arc_lengths = torch.cat([torch.zeros(num_curves, 1, device=device), deltas.cumsum(dim=-1)], dim=-1)
        total_lengths = arc_lengths[:, -1:]
        normalized_lengths = arc_lengths / total_lengths

        # Step 3
        target_lengths = torch.linspace(0, 1, num_samples, device=device)
        target_lengths = target_lengths.unsqueeze(0).expand(normalized_lengths.size(0), -1)
        indices = torch.searchsorted(normalized_lengths, target_lengths) 

        indices = torch.clamp(indices, 1, fine_samples - 1)
        low_indices = indices - 1
        high_indices = indices
        low_lengths = torch.gather(normalized_lengths, 1, low_indices)  # (num_curves, num_samples)
        high_lengths = torch.gather(normalized_lengths, 1, high_indices)  # (num_curves, num_samples)

        low_t = t_values_fine[low_indices]  # (num_curves, num_samples)
        high_t = t_values_fine[high_indices]  # (num_curves, num_samples)

        high_low_diff = high_lengths - low_lengths + 1e-8

        t_values_uniform = low_t + (target_lengths - low_lengths) / high_low_diff * (high_t - low_t)

        t_powers_uniform = t_values_uniform[:, :, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)  # t^i
        one_minus_t_powers_uniform = (1 - t_values_uniform[:, :, None]) ** torch.arange(n, -1, -1, dtype=torch.float32, device=device)  # (1-t)^(n-i)
        bernstein_uniform = comb * one_minus_t_powers_uniform * t_powers_uniform  # (num_curves, num_samples, n+1)
        sampled_points = torch.sum(bernstein_uniform[..., None] * bezier_curves[:, None, :, :], dim=2)  # (num_curves, num_samples, 2)

        bezier_derivative = n * (bezier_curves[:, 1:, :] - bezier_curves[:, :-1, :])  # Derivative control points
        comb_derivative = torch.tensor([math.comb(n - 1, i) for i in range(n)], dtype=torch.float32, device=device)
        bernstein_derivative = comb_derivative * one_minus_t_powers_uniform[:, :, :-1] * t_powers_uniform[:, :, 1:]  # (num_curves, num_samples, n)
        tangents = torch.sum(bernstein_derivative[..., None] * bezier_derivative[:, None, :, :], dim=2)  # Tangents
        normals = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        # print("one curve sample shape: ", sampled_points.shape)
        return sampled_points, normals, tangents

    def compute_adjusted_tangents_from_points(self, points, length=1000.0):
        """
        Compute tangents directly based on point-to-point directions without using existing tangents.

        Parameters:
            points: torch.Tensor
                Shape: (N, H, 2) - sampling points in 2D space.
            length: float
                The desired length of the tangent vectors.
        
        Returns:
            Adjusted tangents: torch.Tensor
                Shape: (N, H, 2) - computed and adjusted tangents.
        """
        directions = points[:, 1:] - points[:, :-1]  # Shape: (N, H-1, 2)
        last_direction = directions[:, -1:] 
        directions = torch.cat([directions, last_direction], dim=1)  # Shape: (N, H, 2)

        norms = torch.norm(directions, dim=-1, keepdim=True)  # Shape: (N, H, 1)
        norms = torch.clamp(norms, min=1e-8)
        directions = directions / norms  # Normalize directions

        N, H, _ = points.shape
        mask = torch.ones((N, H, 1), device=points.device)  # Shape: (N, H, 1)
        midpoint = H // 2
        mask[:, midpoint:] = -1

        tangents = directions * mask * length  # Adjusted tangents
        return tangents

    def calculate_shape_grad(self, points, color_grad):
        tangents = self.compute_adjusted_tangents_from_points(points)
        contrib = torch.abs(color_grad.sum(dim=-1, keepdim=True))
        return tangents * contrib

    def visualize_lines_with_tangents(self, points, tangents, colors):
        tangents = self.compute_adjusted_tangents_from_points(points)
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if isinstance(tangents, torch.Tensor):
            tangents = tangents.detach().cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

        N, H, _ = points.shape

        # Create the plot
        plt.figure(figsize=(10, 10))
        for n in range(N):
            # Extract the nth line
            line_points = points[n]  # Shape: (H, 2)
            line_colors = colors[n]  # Shape: (H, 3)
            
            # Plot the line by connecting points
            # plt.plot(line_points[:, 0], line_points[:, 1], color=line_colors[0], alpha=0.5, label=f"Line {n+1}")
            
            # Scatter points and add tangents
            for i in range(H):
                plt.scatter(line_points[i, 0], line_points[i, 1], color=line_colors[i], s=5)
                plt.arrow(line_points[i, 0], line_points[i, 1],
                        tangents[n, i, 0] , tangents[n, i, 1], 
                        head_width=3, head_length=5, fc=line_colors[i], ec='gray')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Visualization of Lines with Tangents and Colors")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

    def split_bezier_segments(self, ctrl_pts):
        degree = self.bezier_degree + 1
        N, total_pts, _ = ctrl_pts.shape
        assert (total_pts - 1) % degree == 0, "Control points must follow M*D + 1 pattern"
        M = (total_pts - 1) // degree
        segments = [ctrl_pts[:, i*degree:i*degree + degree + 1, :].unsqueeze(1) for i in range(M)]  # list of (N, 1, D+1, 2)
        segments = torch.cat(segments, dim=1)  # shape: (N, M, D+1, 2)
        segments = segments.contiguous().view(-1, degree + 1, 2)  # shape: (N * M, D+1, 2)
        return segments

    def sample_bezier_area(self, control_points, resolution=20, factor=1):
        """
        Samples points along valid line segments formed by Bézier curve intersections with horizontal scanlines.

        Args:
            control_points (torch.Tensor): Shape (num_curves, num_control_points, 2), Bézier control points.
            sample_points (torch.Tensor): Shape (num_curves, num_points, 2), Bézier curve sample points.
            resolution (int): Number of horizontal scanlines.
            total_samples_per_row (int): Total number of points to sample per scanline.

        Returns:
            sampled_positions (torch.Tensor): (num_curves, resolution * total_samples_per_row, 2)
                - Sampled points along the valid segments.
        """
        N, total_pts, _ = control_points.shape
        num_samples = self.num_samples * factor
        assert (total_pts - 2) % 2 == 0, "Control point count must be 2M+2 for degree M Bézier pairs"
        M = (total_pts - 2) // 2  # degree of Bézier
        bezier1 = control_points[:, :M+2, :]
        bezier2 = torch.cat([control_points[:, M+1:, :], control_points[:, 0:1, :]], dim=1).flip(dims=[1])

        bezier1_segments = self.split_bezier_segments(bezier1)
        bezier2_segments = self.split_bezier_segments(bezier2)
        # print("bezier seg: ", bezier1_segments.shape)
        boundary_beziers = torch.cat([bezier1_segments, bezier2_segments], dim=0)  # (2*N*M, degree+2, 2)
        boundary = self.sample_bezier_curves_uniform(boundary_beziers, num_samples)
        
        bezier1_samples, bezier2_samples = boundary.chunk(2, dim=0)
        # print("bezier seg: ", bezier1_samples.shape)        
        sampled_boundary = torch.stack([bezier1_samples.reshape(N, int(self.num_beziers * num_samples / 2), 2), \
            bezier2_samples.reshape(N, int(self.num_beziers * num_samples / 2), 2)], dim=1)


        bezier1 = bezier1.unsqueeze(1)
        bezier2 = bezier2.unsqueeze(1) 
        x = torch.linspace(-2, 2, resolution, device=control_points.device)  
        # cdf_values = dist.Normal(0, 0.85).cdf(x)
        t_vals = torch.linspace(-2, 2, resolution, device=control_points.device)
        t_vals = dist.Normal(0, 0.85).cdf(t_vals).view(1, resolution, 1, 1)  # (1, R, 1, 1)
        # t_vals = dist.Normal(0, 1.0).cdf(t_vals).view(1, resolution, 1, 1)  # (1, R, 1, 1)
        interp_cp = (1 - t_vals) * bezier1 + t_vals * bezier2  # (N, R, M+2, 2)
        interp_cp_flat = interp_cp.view(-1, M + 2, 2)  # (N * R, M+2, 2)

        interp_segments = self.split_bezier_segments(interp_cp_flat)
        interp_samples = self.sample_bezier_curves_uniform(interp_segments, num_samples)  # (N * R, num_samples, 2)
        # print("result: ", sampled_boundary.shape, interp_samples.shape)
        return sampled_boundary, interp_samples.view(self._control_points.shape[0], resolution, -1, 2).detach()

    def bezier_same_side_mask(self, bezier_curves: torch.Tensor) -> torch.Tensor:
        """
        Given a set of closed curves (Nx8x2), determine if both Bézier curves 
        are on the same side of the line formed by control points 0 and 4.

        Args:
            bezier_curves (torch.Tensor): Shape (N, 8, 2), representing N closed curves 
                                        formed by two Bézier segments.

        Returns:
            torch.Tensor: Boolean mask of shape (N,), where True means curves are on the same side.
        """
        # Extract key points
        P0 = bezier_curves[:, 0]  # (N, 2)
        P4 = bezier_curves[:, 4]  # (N, 2)

        # Compute direction vector of line P0-P4
        line_vec = P4 - P0  # (N, 2)

        # Select the control points of both Bézier curves (excluding P0 and P4)
        control_points = torch.cat([bezier_curves[:, 1:4], bezier_curves[:, 5:8]], dim=1)  # (N, 6, 2)

        # Compute vectors from P0 to each control point
        vecs_to_points = control_points - P0[:, None, :]  # (N, 6, 2)

        # Compute cross product to determine relative position
        cross_products = line_vec[:, 0:1] * vecs_to_points[:, :, 1] - line_vec[:, 1:2] * vecs_to_points[:, :, 0]  # (N, 6)

        # Check if all cross products have the same sign for each Bézier curve
        # same_side_mask = (cross_products > 0).all(dim=1) | (cross_products < 0).all(dim=1)
        same_side_mask = ((cross_products > 0).sum(dim=1) > 5) | ((cross_products < 0).all(dim=1) > 5)
        return same_side_mask

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # print("into device: ", group["params"][0].device, mask.device, group["name"])
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                stored_state["exp_avg_diff"] = stored_state["exp_avg_diff"][mask]
                stored_state["neg_pre_grad"] = stored_state["neg_pre_grad"][mask]

                del self.optimizer.state[group['params'][0]]
                if group["name"] == "xyz":
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                else:
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                # optimizable_tensors[group["name"]] = group["params"][0]
                opacity_mask = mask.detach().cpu()
                # print("opacity device is: ", group["params"][0].device, mask.device)
                group["params"][0] = nn.Parameter(group["params"][0][opacity_mask].requires_grad_(False))
                optimizable_tensors[group["name"]] = group["params"][0].to(mask.device)
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # print("state :", group["name"], stored_state["exp_avg"].shape, torch.zeros_like(extension_tensor).shape)
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_diff"] = torch.cat((stored_state["exp_avg_diff"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["neg_pre_grad"] = torch.cat((stored_state["neg_pre_grad"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # print("state :", group["name"],group["params"][0].device, extension_tensor.to(group["params"][0].device))
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor.to(group["params"][0].device)), dim=0).requires_grad_(False))
                optimizable_tensors[group["name"]] = group["params"][0].to(extension_tensor.device)
        return optimizable_tensors

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # print("reopacity: ", name, tensor)
                # stored_state = self.optimizer.state.get(group['params'][0], None)
                # stored_state["exp_avg"] = torch.zeros_like(tensor)
                # stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                # stored_state["exp_avg_diff"] = torch.cat((stored_state["exp_avg_diff"], torch.zeros_like(tensor)), dim=0)
                # stored_state["neg_pre_grad"] = torch.cat((stored_state["neg_pre_grad"], torch.zeros_like(tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                # self.optimizer.state[group['params'][0]] = stored_state
                # print("reopacity: ", group["name"], group["params"][0])
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_beizer_curves(self, mask):
        valid_beizer_mask = mask
        optimizable_tensors = self._prune_optimizer(valid_beizer_mask)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        # self._rotation = optimizable_tensors["rotation"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]
        self._depth = optimizable_tensors["depth"]
        # self._grad = self._grad[mask]
        self.xyz=self.xyz.view(-1,self.total_num_sample,2)[mask].view((-1,self.total_num_sample,2))

    def densification_postfix(self, new_control_points, new_features, new_cholesky, new_depth, new_opacities, new_scaling):
        d = {"control_points": new_control_points,
        "features_dc": new_features,
        "cholesky": new_cholesky,
        "opacity": new_opacities,
        "scaling": new_scaling,
        "depth" : new_depth}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # print("optimizer: ", optimizable_tensors["control_points"].shape)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._depth = optimizable_tensors["depth"]

    def modified_control_points(self, bezier_curves):
        
        N, M, _ = bezier_curves.shape
        P0 = bezier_curves[:, 0, :].unsqueeze(1)    # (N, 1, 2)
        P_last = bezier_curves[:, -1, :].unsqueeze(1) # (N, 1, 2)
        
        t = torch.linspace(0, 1, steps=M, device=bezier_curves.device).view(1, M, 1)
        new_bezier_curves = P0 + t * (P_last - P0)
        
        return new_bezier_curves

    def split_condition_1(self, num, N=2):
        opacities = torch.sigmoid(self._opacity)
        start = opacities[:, 0]
        end = opacities[:, -1]
        middle1 = opacities[:, 0:-1].mean(dim=-1)
        # middle2 = opacities[:, 2]

        candidate1 = torch.abs(torch.max(start, end) - middle1)
        candidate2 = torch.abs(middle1 - torch.min(start, end))
        max_gap = torch.maximum(candidate1, candidate2)
        max_gap_flat = max_gap.view(-1)

        diffs = torch.abs(start - end)
        diffs_flat = diffs.view(-1)

        combined = torch.maximum(diffs, max_gap)
        topk_values, topk_indices = torch.topk(combined, k=num)
        selected_pts_mask = torch.zeros_like(diffs, dtype=torch.bool)
        selected_pts_mask.view(-1)[topk_indices] = True
        xyz = self.xyz.view(-1,self.total_num_sample,2)
        if selected_pts_mask.sum() > 0:
            new_features = self._features_dc[selected_pts_mask].repeat(N,1)
            new_cholesky = self._cholesky[selected_pts_mask].repeat(N,1) / 2
            new_control_points = self._control_points[selected_pts_mask].clone()
            new_control_points_split = self._control_points[selected_pts_mask].clone()
            new_depth = self._depth[selected_pts_mask].repeat(N,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            new_scaling = self._scaling[selected_pts_mask]
            new_scaling[:, 0] = new_scaling[:, 0] / 2
            new_scaling =new_scaling.repeat(N,1)
            
            filtered_opacity = self._opacity[selected_pts_mask]
            K = filtered_opacity.shape[0]

            # print("opacity: ", new_opacity[0:selected_pts_mask.shape[0], 0:2].shape, self._opacity[selected_pts_mask][0].shape)
            new_opacity[0:K, 0:2] = filtered_opacity[:, 0].unsqueeze(1).repeat(K, 2)
            new_opacity[0:K, 2:] = filtered_opacity[:, 1].unsqueeze(1)
            new_opacity[K:, 0:1] = filtered_opacity[:, 1].unsqueeze(1)
            new_opacity[K:, 1:] = filtered_opacity[:, 2].unsqueeze(1).repeat(K, 2)

            xyz = xyz[selected_pts_mask]
            # print("xyz: ", xyz.shape, )
            new_control_points[:, -1, :] = xyz[:, int(self.num_samples / 2) - 1, :]
            new_control_points_split[:, 0, :] = xyz[:, int(self.num_samples / 2) + 1, :]
            new_control_points = self.modified_control_points(new_control_points)
            new_control_points_split = self.modified_control_points(new_control_points_split)
            # print("the changed output is: ", new_control_points[0])
            new_control_points = torch.cat((new_control_points, new_control_points_split), dim=0)
            self.prune_beizer_curves(~selected_pts_mask)
            self.densification_postfix(new_control_points, new_features, new_cholesky, new_depth, new_opacity, new_scaling)

    def densify(self, num, pos_init_method, gt_image, radii=0.02):
        # print("gt image shape: ", gt_image.shape, num)
        centers = torch.tensor([pos_init_method() for _ in range(num)], dtype=torch.float32).to(self._control_points.device)
        # print("centers: ",centers.shape, centers[:, 0].max(), centers[:, 1].max())
        centers_rounded = centers.round().long()
        centers_rounded[:, 0] = torch.clamp(centers_rounded[:, 0], 0, self.H - 1)
        centers_rounded[:, 1] = torch.clamp(centers_rounded[:, 1], 0, self.W - 1)
        gt_values = gt_image[:, :, centers_rounded[:, 0], centers_rounded[:, 1]].squeeze(0).T
        # print("gt values: ", gt_values.shape, self._control_points.max())
        new_control_points = self._initialize_control_points_with_center(centers.unsqueeze(1), radii).to(self._control_points.device)
        new_cholesky = nn.Parameter(torch.rand(centers.shape[0], 3)).to(self._control_points.device)
        # new_features = nn.Parameter(torch.rand(centers.shape[0], 3)).to(self._control_points.device)
        logits = torch.logit(gt_values.clamp(1e-6, 1 - 1e-6))  # 防止除以0或log(0)
        new_features = nn.Parameter(logits.to(self._control_points.device))
        # new_features = nn.Parameter(gt_values.to(self._control_points.device))
        new_depth = nn.Parameter(torch.zeros(centers.shape[0], 1)).to(self._control_points.device)
        new_scaling = nn.Parameter(torch.rand(centers.shape[0], 1)).to(self._control_points.device)
        new_opacity = nn.Parameter(torch.ones(centers.shape[0], self._opacity.shape[1])).to(self._control_points.device)
        self.densification_postfix(new_control_points, new_features, new_cholesky, new_depth, new_opacity, new_scaling)

    def densify_and_split(self, num, grad_threshold=2e-6, N=2):
        opacities = torch.sigmoid(self._opacity)
        start_end = torch.min(opacities[:, 0], opacities[:, 3])
        middle = torch.min(opacities[:, 1], opacities[:, 2])
        
        grad = torch.abs(self._features_dc.grad.sum(-1)) / (opacities.sum(-1) / 4)

        diffs = torch.abs(start_end - middle)
        diffs_mask = diffs > 0.3

        diffs_abs = torch.abs(start_end - middle)
        score = diffs_abs * grad
        score_flat = score.view(-1)

        topk_values, topk_indices = torch.topk(score_flat, k=num)
        selected_pts_mask = torch.zeros_like(score, dtype=torch.bool)
        selected_pts_mask.view(-1)[topk_indices] = True
        xyz = self.xyz.view(-1,self.num_samples,2)
        if selected_pts_mask.sum() > 0:
            new_features = self._features_dc[selected_pts_mask].repeat(N,1)
            new_cholesky = self._cholesky[selected_pts_mask].repeat(N,1) / 2
            new_control_points = self._control_points[selected_pts_mask].clone()
            new_control_points_split = self._control_points[selected_pts_mask].clone()
            new_depth = self._depth[selected_pts_mask].repeat(N,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            new_scaling = self._scaling[selected_pts_mask]
            new_scaling[:, 1] = new_scaling[:, 1] / 2
            new_scaling =new_scaling.repeat(N,1)
            
            filtered_opacity = self._opacity[selected_pts_mask]
            K = filtered_opacity.shape[0]

            # print("opacity: ", new_opacity[0:selected_pts_mask.shape[0], 0:2].shape, self._opacity[selected_pts_mask][0].shape)
            new_opacity[0:K, 0:2] = filtered_opacity[:, 0].unsqueeze(1).repeat(K, 2)
            new_opacity[0:K, 2:] = filtered_opacity[:, 1].unsqueeze(1).repeat(K, 2)
            new_opacity[K:, 0:2] = filtered_opacity[:, 2].unsqueeze(1).repeat(K, 2)
            new_opacity[K:, 2:] = filtered_opacity[:, 3].unsqueeze(1).repeat(K, 2)

            xyz = xyz[selected_pts_mask]
            # print("xyz: ", xyz.shape, )
            new_control_points[:, -1, :] = xyz[:, int(self.num_samples / 2) - 1, :]
            new_control_points_split[:, 0, :] = xyz[:, int(self.num_samples / 2) + 1, :]
            new_control_points = self.modified_control_points(new_control_points)
            new_control_points_split = self.modified_control_points(new_control_points_split)
            # print("the changed output is: ", new_control_points[0])
            new_control_points = torch.cat((new_control_points, new_control_points_split), dim=0)
            self.prune_beizer_curves(~selected_pts_mask)
            self.densification_postfix(new_control_points, new_features, new_cholesky, new_depth, new_opacity, new_scaling)


    # def cutting_edges(self):
    #     Line_grad = (self.xyz_gradient_accum / self.denom).view(self.num_curves, -1, 2)
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(torch.sigmoid(self._opacity), torch.ones_like(torch.sigmoid(self._opacity))*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # print("optimizable is: ", optimizable_tensors)
        self._opacity = optimizable_tensors["opacity"]
        # print("self.opacity.shape", self._opacity.shape)

    def calculate_edge_grad(self):
        self.visualize_lines_with_tangents(self.xys.view(self.num_curves,-1,2)[0:1,:,:], self._tangents[0:1,:,:], self.features_dc.view(self.num_curves,-1,3)[0:1,:,:])
        self.xys.grad += self._tangents.view(-1, 2)
        print("color grad and tangents: ", self.features_dc.grad.shape, self.xys.view(self.num_curves,-1,2).shape, self._tangents.shape)

    def forward(self, factor=1, denser_sample=False):
        final_h = self.H * factor
        final_w = self.W * factor
        self.xyz, self.xyz_area = self.get_xyz_and_depth(factor, denser_sample)
        if self.mode == 'closed':
            xyz_input = torch.cat([self.xyz, self.xyz_area], dim=1).contiguous().view(-1, 2)
        else:
            xyz_input = self.xyz
        
        with torch.no_grad():
            rotation_input = self.compute_rotations(
                    self.xyz.view(self._control_points.shape[0], -1, 2)).view(-1, 1).detach()    
            scaling = self.get_scaling(factor)

        opacity = self.get_opacity
        features_dc = self.get_features.contiguous()
        # self.xys, depths_, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xyz_input, scaling, rotation_input, self.H, self.W, self.tile_bounds)
        self.tile_bounds = (
            (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
            (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        self.xys, depths_, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xyz_input, scaling, rotation_input, final_h, final_w, self.tile_bounds)

        depth = self.get_depth.detach()
        if self.mode == "closed":
            with torch.no_grad():
                boxes = self.compute_aabb(self.xyz.view(self.xyz.shape[0], -1, 2))
                ratio = self.W / self.H
                # print("ratio: ", ratio)
                widths = (boxes[:, 2] - boxes[:, 0]) * ratio
                heights = (boxes[:, 3] - boxes[:, 1])
                depth = widths * heights
                self._depth.copy_(depth.unsqueeze(-1).contiguous())
            depth = self.get_depth.view(-1, 1)
        else:
            with torch.no_grad():
                xys = self.xys.clone().detach().view(self._control_points.shape[0], -1, 2)
                diffs = torch.norm(xys[:, 1:, :] - xys[:, :-1, :], dim=-1).sum(-1, keepdim=True) * torch.abs(self._scaling.detach())
                self._depth.copy_(diffs)
            depth = self.get_depth.view(-1, 1)

        # print("self.xyz: ", self.xys.shape, depth.shape, self.radii.shape, conics.shape, features_dc.shape, opacity.shape)
        # out_img = rasterize_gaussians(self.xys, depth.detach(), self.radii, conics, num_tiles_hit,
        #         features_dc.view(-1, 3), opacity.view(-1, 1), self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        
        # final_h = self.H * factor
        # final_w = self.W * factor
        if factor > 1:
            print("See the range: ", self.xys.max(), self.xys.min(), self.xyz.max(), self.xyz.min())
            
        out_img = rasterize_gaussians(
                self.xys,
                depth,
                self.radii,
                conics,
                num_tiles_hit,
                features_dc.view(-1, 3),
                opacity.view(-1, 1),
                final_h, final_w,
                self.BLOCK_H, self.BLOCK_W,
                background=self.background,
                return_alpha=False)

        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, final_h, final_w, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    def forward_area_boundary(self):
        self.xyz, self.xyz_area = time_cuda(self.get_xyz_and_depth, "get_xyz_and_depth")
        rotation_input = time_cuda(
            lambda: self.compute_rotations(
                self.xyz.view(self._control_points.shape[0], -1, 2)
            ).view(-1, 1).detach(),
            "compute_rotations"
        )
        
        xyz_input = torch.cat([self.xyz, self.xyz_area], dim=1).contiguous().view(-1, 2)
        B, S1, P, _ = self.xyz.shape   # B = 1024, S1 = 2, P = 32
        S2 = self.xyz_area.shape[1]    # S2 = 20
        total_S = S1 + S2              # 22

        b_idx = torch.arange(B, device=self.xyz.device).view(-1, 1, 1)       # (B, 1, 1)
        s_idx = torch.arange(S1, device=self.xyz.device).view(1, -1, 1)      # (1, S1, 1)
        p_idx = torch.arange(P, device=self.xyz.device).view(1, 1, -1)       # (1, 1, P)

        # compute the flattened index
        xyz_indices = (b_idx * total_S * P + s_idx * P + p_idx).reshape(-1)  # shape: (B * S1 * P,)

        scaling = self.get_scaling(factor=1)
        opacity = self.get_opacity
        features_dc = self.get_features.contiguous()
        # self.xys, depths_, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(xyz_input, scaling, rotation_input, self.H, self.W, self.tile_bounds)
        self.xys, depths_, self.radii, conics, num_tiles_hit = time_cuda(
            lambda: project_gaussians_2d_scale_rot(
                xyz_input[xyz_indices], scaling[xyz_indices], rotation_input[xyz_indices],
                self.H, self.W, self.tile_bounds
            ),
            "project_gaussians_2d_scale_rot"
        )
        depth = self.get_depth.detach()
        if self.mode == "closed":
            boxes = self.compute_aabb(self.xyz.view(self.xyz.shape[0], -1, 2))
            ratio = self.W / self.H
            # print("ratio: ", ratio)
            widths = (boxes[:, 2] - boxes[:, 0]) * ratio
            heights = (boxes[:, 3] - boxes[:, 1])
            depth = widths * heights
            with torch.no_grad():
                # print("self.depth", self._depth.shape, depth.shape)
                self._depth.copy_(depth.unsqueeze(-1).contiguous())
            depth = self.get_depth.view(-1, 1)[xyz_indices]
            opacity = opacity[xyz_indices]
            features_dc = features_dc.view(-1, 3)[xyz_indices]
        
        out_img = time_cuda(
            lambda: rasterize_gaussians(
                self.xys,
                depth,
                self.radii,
                conics,
                num_tiles_hit,
                features_dc.view(-1, 3),
                opacity.view(-1, 1),
                self.H, self.W,
                self.BLOCK_H, self.BLOCK_W,
                background=self.background,
                return_alpha=False
            ),
            "rasterize_gaussians"
        )
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}



    def project_point_to_line(self, P, A, B):
        AB = B - A
        AP = P - A
        AB_unit = AB / (torch.norm(AB, dim=-1, keepdim=True) + 1e-8)
        return torch.sum(AP * AB_unit, dim=-1)
    
    def vector_cross(self, A, B):
        return A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]

    def shape_alignment(self):
        control_points = self._control_points.clone().detach()
        control_points_fixed = control_points.clone()
        D = control_points[:, 4] - control_points[:, 0]

        proj_P2 = self.project_point_to_line(control_points[:, 1], control_points[:, 0], control_points[:, 4])
        proj_P3 = self.project_point_to_line(control_points[:, 2], control_points[:, 0], control_points[:, 4])
        proj_P4 = self.project_point_to_line(control_points[:, 3], control_points[:, 0], control_points[:, 4])
        
        swap_mask_1 = proj_P3 < proj_P2
        if swap_mask_1.any():
            temp = control_points_fixed[swap_mask_1, 1].clone()
            control_points_fixed[swap_mask_1, 1] = control_points_fixed[swap_mask_1, 2]
            control_points_fixed[swap_mask_1, 2] = temp
        
        swap_mask_2 = proj_P4 < proj_P3
        if swap_mask_1.any():
            temp = control_points_fixed[swap_mask_2, 2].clone()
            control_points_fixed[swap_mask_2, 2] = control_points_fixed[swap_mask_2, 3]
            control_points_fixed[swap_mask_2, 3] = temp

        proj_P5 = self.project_point_to_line(control_points[:, 5], control_points[:, 0], control_points[:, 4])
        proj_P6 = self.project_point_to_line(control_points[:, 6], control_points[:, 0], control_points[:, 4])
        proj_P7 = self.project_point_to_line(control_points[:, 7], control_points[:, 0], control_points[:, 4])
        swap_mask_3 = proj_P5 > proj_P6
        if swap_mask_3.any():
            temp = control_points_fixed[swap_mask_3, 5].clone()
            control_points_fixed[swap_mask_3, 5] = control_points_fixed[swap_mask_3, 6]
            control_points_fixed[swap_mask_3, 6] = temp
        
        swap_mask_4 = proj_P6 > proj_P7
        if swap_mask_4.any():
            temp = control_points_fixed[swap_mask_4, 6].clone()
            control_points_fixed[swap_mask_4, 6] = control_points_fixed[swap_mask_4, 7]
            control_points_fixed[swap_mask_4, 7] = temp
        with torch.no_grad():
            self._control_points.copy_(control_points_fixed)

    def shape_refinement(self, threshold=0.1):
        for i in range(self.num_beziers):
            BA_vec = self._control_points[:,3*i+1,:] - self._control_points[:,3*i+0,:]
            CD_vec = self._control_points[:,3*i+2,:] - self._control_points[:,3*i+3,:]
            BC_vec = self._control_points[:,3*i+1,:] - self._control_points[:,3*i+2,:]
            CB_vec = self._control_points[:,3*i+2,:] - self._control_points[:,3*i+1,:]

            dot = (BA_vec * CD_vec).sum(dim=1)  # shape: (N,)
            eps = 1e-8
            norm_v1 = BA_vec.norm(dim=1) + eps  # shape: (N,)
            norm_v2 = CD_vec.norm(dim=1) + eps  # shape: (N,)

            norm_mask = norm_v1 >= norm_v2 
            cos_theta = dot / (norm_v1 * norm_v2)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angles = torch.acos(cos_theta)  # shape: (N,)

            threshold = 0.1
            threshold_turn = 0.1
            mask = angles < threshold  # mask 的 shape 为 (N,), 类型为 bool
            # print("mask sum: ", mask.sum())
            new_beizers = self._control_points.clone()
            # new_beizers_v1 = 
            new_beizers[mask & norm_mask][:, [3*i+0,3*i+3], :] = self._control_points[(mask & norm_mask)][:, [3*i+0, 3*i+3], :]
            # new_beizers[mask & ~norm_mask][:, [0,3], :] = self._control_points[(mask & ~norm_mask)][:, [2, 3], :]
            new_beizers[mask & norm_mask][:, [3*i+0,3*i+3], :] = self._control_points[(mask & norm_mask)][:, [3*i+2, 3*i+3], :]
            new_beizers[mask] = self.modified_control_points(new_beizers[mask])
        with torch.no_grad():
            self._control_points.copy_(new_beizers)

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        if self.num_beziers > 2:
            loss_reg = bezier_shape_regularizer(self._control_points, self.bezier_degree * int(self.num_beziers / 2) + 1)
        else:
            loss_reg = bezier_shape_regularizer(self._control_points, self.bezier_degree)
        loss += loss_reg
        loss += 1e-2 * torch.abs(torch.sigmoid(self._opacity) - 1.0).mean()
        # print("loss reg: ", loss_reg)
        # loss += xing_loss(self._control_points, scale=1e-2)

        # # print("boundary: ", self.xyz.shape)
        loss += curvature_loss(self.xyz, self.num_samples)
        loss += boundary_loss_on_joints(self._control_points, self.bezier_degree + 1)
        loss.backward()
        # time_cuda(
        #     lambda: loss.backward(),
        #     "backward"
        # )  # Shape: (num_curves, num_beziers, 4, 2)
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.iter += 1
        self.scheduler.step()
        return loss, psnr, image