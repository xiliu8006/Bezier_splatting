import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.draw import line_aa
import numpy as np
import cv2
import torch.optim as optim
from scipy.special import comb
import os

class SVGRenderer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bezier_curves, distances_, rendered_image, num_samples=50):
        """
            Move Bézier curve samples along their normals by given distances.
            
            Args:
            - bezier_curves_samples: Tensor of shape (N, 4, 2), Bézier curve sampling points.
            - distance: Tensor of shape (N, 4, 2), distances to move along the normals.
                        distance[:,:,0] is for moving in the positive normal direction.
                        distance[:,:,1] is for moving in the negative normal direction.
            
            Returns:
            - moved_positive: Tensor of shape (N, 4, 2), samples moved along the positive normal direction.
            - moved_negative: Tensor of shape (N, 4, 2), samples moved along the negative normal direction.
        """
        rendered_image = rendered_image.squeeze(0)
        W = rendered_image.shape[-1]
        distances = distances_.unsqueeze(1).unsqueeze(2).repeat(1, num_samples, 2)
        # print("bezier_curves: ", bezier_curves.max())
        bezier_curves_samples, normals = SVGRenderer.sample_bezier_curves(bezier_curves, num_samples)
        ctx.save_for_backward(bezier_curves_samples, normals, rendered_image)
        # print("distances shape: ", distances.shape, bezier_curves_samples.shape, normals.shape)
        # boundary_samples_positive = bezier_curves_samples + normals * distances[:, :, :1]
        # boundary_samples_negative = bezier_curves_samples - normals * distances[:, :, -1:] 
        # ctx.save_for_backward(bezier_curves_samples, normals, boundary_samples_positive, boundary_samples_negative, rendered_image)
        return rendered_image

    # @staticmethod
    # def backward(ctx, grad_output):
    #     # print("grad_output: ", grad_output.shape, grad_output.max(), grad_output.min())
    #     bezier_curves_samples, normals, boundary_samples_positive, boundary_samples_negative, rendered_image = ctx.saved_tensors
    #     contrib_pos = SVGRenderer.cal_contrib(rendered_image, boundary_samples_positive, normals, grad_output)
    #     contrib_neg = SVGRenderer.cal_contrib(rendered_image, boundary_samples_negative, -normals, grad_output)
    #     # print("contrib_neg is: ", contrib_neg.shape, bezier_curves_samples.shape)
    #     grad_distance = torch.zeros_like(boundary_samples_positive)
    #     grad_distance[:,:, 0] = contrib_pos
    #     grad_distance[:,:, 1] = contrib_neg
    #     # print("contrib: ",contrib_pos.mean())
    #     grad_distance = grad_distance.sum(dim=1).sum(dim=1)

    #     grad_bezier_curves = SVGRenderer.bezier_gradient(contrib_pos, normals, bezier_curves_samples.shape[1], bezier_curves_samples.device)
    #     grad_bezier_curves += SVGRenderer.bezier_gradient(contrib_neg, -normals, bezier_curves_samples.shape[1], bezier_curves_samples.device)
    #     # print("backward gradient: ", grad_bezier_curves.mean(), grad_distance.mean())
    #     return grad_bezier_curves, grad_distance, None, None
    
    @staticmethod
    def backward(ctx, grad_output):
        # print("grad_output: ", grad_output.shape, grad_output.max(), grad_output.min())
        bezier_curves_samples, normals, rendered_image = ctx.saved_tensors
        contrib_pos = SVGRenderer.cal_contrib(rendered_image, boundary_samples_positive, normals, grad_output)
        contrib_neg = SVGRenderer.cal_contrib(rendered_image, boundary_samples_negative, -normals, grad_output)
        # print("contrib_neg is: ", contrib_neg.shape, bezier_curves_samples.shape)
        grad_distance = torch.zeros_like(boundary_samples_positive)
        grad_distance[:,:, 0] = contrib_pos
        grad_distance[:,:, 1] = contrib_neg
        # print("contrib: ",contrib_pos.mean())
        grad_distance = grad_distance.sum(dim=1).sum(dim=1)

        grad_bezier_curves = SVGRenderer.bezier_gradient(contrib_pos, normals, bezier_curves_samples.shape[1], bezier_curves_samples.device)
        grad_bezier_curves += SVGRenderer.bezier_gradient(contrib_neg, -normals, bezier_curves_samples.shape[1], bezier_curves_samples.device)
        # print("backward gradient: ", grad_bezier_curves.mean(), grad_distance.mean())
        return grad_bezier_curves, grad_distance, None, None


    @staticmethod
    def bezier_gradient(contrib, normal, num_samples, device='cpu'):
        """
        并行化计算采样点梯度对贝塞尔曲线控制点的影响 (PyTorch 实现)。
        
        Args:
            contrib (torch.Tensor): 形状为 (num_samples, 4)，每个采样点的贡献值。
            normal (torch.Tensor): 形状为 (num_samples, 4, 2)，每个采样点的法向量。
            num_samples (int): 采样点的数量。
            device (str): 设备类型（如 'cpu' 或 'cuda'）。
        
        Returns:
            torch.Tensor: 更新后的控制点梯度，形状为 (num_samples, 4, 2)。
        """
        # print("bezier gradient shape: ", contrib.shape, normal.shape, num_samples)
        device = contrib.device  # 获取当前设备
        t_values = torch.linspace(0, 1, num_samples, device=device)  # t 参数值

        # Bernstein 基函数矩阵，形状为 (num_samples, 4)
        B = torch.stack([
            (1 - t_values).pow(3),              # B_0(t)
            3 * (1 - t_values).pow(2) * t_values,  # B_1(t)
            3 * (1 - t_values) * t_values.pow(2),  # B_2(t)
            t_values.pow(3)                     # B_3(t)
        ], dim=-1)  # Shape: (num_samples, 4)

        # 将 B 扩展维度以匹配批处理，形状为 (1, num_samples, 4)
        B = B.unsqueeze(0)  # 扩展到 (1, num_samples, 4)

        # 广播 Bernstein 基函数矩阵，与 contrib 和 normal 一起计算
        grad = (B.unsqueeze(-1) * normal.unsqueeze(2)) * contrib.unsqueeze(-1).unsqueeze(-1)

        # 按采样点累积梯度到控制点，sum over samples
        d_p = grad.sum(dim=1)  # Shape: (N, 4, 2)
        # print("dp translate is: ", d_p)
        return d_p
        
    @staticmethod
    def cal_contrib(image, points, normals, grad_output):
        """
        Sample colors for given points with normals.
        
        Args:
        - image: Tensor of shape (C, H, W), the input image.
        - points: Tensor of shape (N, 4, 2), the (x, y) positions to sample.
        - normals: Tensor of shape (N, 4, 2), the normals at each point.
        
        Returns:
        - sampled_colors: Tensor of shape (N, 4, 3, C), the sampled colors:
            - [:, :, 0, :] - Colors at the nearest integer point (normal-restricted).
            - [:, :, 1, :] - Colors at the first offset point.
            - [:, :, 2, :] - Colors at the second offset point.
        """
        device = points.device

        C, H, W = image.shape
        N, num_points, _ = points.shape  # N x 4 x 2
        # print("points shape: ", points[:5])
        points[:, :, 0] = points[:, :, 0] * W
        points[:, :, 1] = points[:, :, 1] * H

        # Compute nearest integer points (normal-restricted)
        offset = normals / torch.norm(normals, dim=-1, keepdim=True)  # Normalize normals
        # print("offset: ", abs(offset).mean())
        restricted_points = points - offset  # Move along the negative normal direction
        restricted_points_int = restricted_points.round()
        mask = (restricted_points_int[..., 1] >= 0) & (restricted_points_int[..., 1] < H) & \
               (restricted_points_int[..., 0] >= 0) & (restricted_points_int[..., 0] < W)

        restricted_points_int[..., 0] = restricted_points_int[..., 0].clamp(0, W - 1)  # 限制 x 坐标
        restricted_points_int[..., 1] = restricted_points_int[..., 1].clamp(0, H - 1)  # 限制 y 坐标
        
        # Extract x, y indices for integer points
        x0 = restricted_points_int[..., 0].long()
        y0 = restricted_points_int[..., 1].long()
        # print("restricted points: ", restricted_points_int)
        # print(x0.device, y0.device, normals.device, restricted_points_int.device, image.device)
        # Sample colors at integer points
        colors_in = image[:, y0, x0].permute(1, 2, 0)  # (N, 4, C)
        # colors_in[~mask.unsqueeze(-1).expand_as(colors_in)] = 0
        d_colors = grad_output[:, y0, x0].permute(1, 2, 0)
        
        # print("d color: ", d_colors.max(), d_colors.min(), grad_output.shape, grad_output.max(), grad_output.min())
        # Compute offset points
        offset_point1 = restricted_points + 2 * offset

        # 将点取整并限制范围
        offset_point1 = offset_point1.round()  # 四舍五入取整
        offset_point1[..., 0] = offset_point1[..., 0].clamp(0, W - 1)  # 限制 x 坐标在 [0, W-1]
        offset_point1[..., 1] = offset_point1[..., 1].clamp(0, H - 1)  # 限制 y 坐标在 [0, H-1]
        # offset_point2 = restricted_points - 4 * offset  # Second offset point
        # Sample colors at offset points
        x1 = offset_point1[..., 0].long()
        y1 = offset_point1[..., 1].long()

        colors_out = image[:, y1, x1].permute(1, 2, 0)  # Shape: (N, 4, C)
        # colors_out[~mask.unsqueeze(-1).expand_as(colors_out)] = 0
        print("mask shape: ", mask.shape, colors_in.shape, mask.sum())
        # print("offset mean: ", colors_in[:5], colors_out[:5], restricted_points_int[:5], abs(20 * offset).mean())

        contrib = ((colors_in - colors_out) * d_colors).sum(dim=2) / d_colors.shape[1]
        # assert len(contrib) == len(mask), f'contrib shape is {contrib.shape}, mask shape is : {mask.shape}'
        contrib[~mask]=-1e-8
        return contrib

    
    @staticmethod
    def sample_bezier_curves(bezier_curves, num_samples):
        """
        Sample points from multiple Bézier curves and return a tensor of sampled points and normals.
        
        Args:
        - bezier_curves: A tensor of shape (num_curves, num_control_points, 2), where 
                        num_control_points = n + 1 for n-degree Bézier curves.
        - num_samples: Number of points to sample along each curve.
        
        Returns:
        - sampled_points: A tensor of shape (num_curves, num_samples, 2) containing the sampled points.
        - normals: A tensor of shape (num_curves, num_samples, 2) containing the normal vectors.
        """
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("Control points must be 2D coordinates.")

        device = bezier_curves.device

        # Degree of the Bézier curve
        n = num_control_points - 1

        # Parameter values for sampling
        t_values = torch.linspace(0, 1, num_samples, device=device)  # Shape: (num_samples,)

        # Compute Bernstein basis: comb(n, i) * (1 - t)^(n - i) * t^i
        # Compute the binomial coefficients (comb(n, i))
        comb = torch.tensor([math.comb(n, i) for i in range(n + 1)], dtype=torch.float32, device=device)  # (n + 1,)

        # Compute powers of t and (1 - t)
        t_powers = torch.pow(t_values[:, None], torch.arange(n + 1, dtype=torch.float32, device=device))  # (num_samples, n + 1)
        one_minus_t_powers = torch.pow(1 - t_values[:, None], torch.arange(n, -1, -1, dtype=torch.float32, device=device))  # (num_samples, n + 1)

        # Bernstein basis functions
        bernstein = comb * one_minus_t_powers * t_powers  # Shape: (num_samples, n + 1)

        # Expand dimensions to align with bezier_curves for broadcasting
        bernstein = bernstein[None, :, :]  # Shape: (1, num_samples, n + 1)

        # Sample points using weighted sum of control points
        sampled_points = torch.sum(bernstein[..., None] * bezier_curves[:, None, :, :], dim=2)  # Shape: (num_curves, num_samples, 2)

        # Compute derivatives for tangent vectors
        comb_derivative = torch.tensor([math.comb(n - 1, i) for i in range(n)], dtype=torch.float32, device=device)  # (n,)
        t_powers_derivative = torch.pow(t_values[:, None], torch.arange(n, dtype=torch.float32, device=device))  # (num_samples, n)
        one_minus_t_powers_derivative = torch.pow(1 - t_values[:, None], torch.arange(n - 1, -1, -1, dtype=torch.float32, device=device))  # (num_samples, n)
        bernstein_derivative = comb_derivative * one_minus_t_powers_derivative * t_powers_derivative  # Shape: (num_samples, n)

        # Adjust control points for derivative
        bezier_derivative = n * (bezier_curves[:, 1:, :] - bezier_curves[:, :-1, :])  # Shape: (num_curves, n, 2)

        # Expand dimensions to align with bezier_derivative for broadcasting
        bernstein_derivative = bernstein_derivative[None, :, :]  # Shape: (1, num_samples, n)

        # Compute tangent vectors
        tangents = torch.sum(bernstein_derivative[..., None] * bezier_derivative[:, None, :, :], dim=2)  # Shape: (num_curves, num_samples, 2)

        # Normalize tangents to compute normals
        normals = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)  # Rotate tangents 90 degrees
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)  # Normalize normals
        # print("sample_points: ", sampled_points.max())
        return sampled_points, normals


def bernstein_poly(i, n, t):
    """
    Compute the Bernstein basis polynomial of n, i as a function of t.
    """
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve(control_points, num_samples=100):
    """
    Compute points on a Bézier curve given control points using matrix operations for efficiency.

    Args:
    - control_points: A NumPy array of shape (n+1, 2), where n is the degree of the Bézier curve.
    - num_samples: Number of points to sample along the curve.

    Returns:
    - curve: A NumPy array of shape (num_samples, 2), representing the sampled points.
    """
    n = len(control_points) - 1  # Degree of the Bézier curve
    t = np.linspace(0, 1, num_samples)  # (num_samples,)
    
    # Compute Bernstein basis functions as a matrix
    t_powers = np.power(t[:, None], np.arange(n + 1))  # (num_samples, n+1)
    one_minus_t_powers = np.power(1 - t[:, None], np.arange(n, -1, -1))  # (num_samples, n+1)
    bernstein_matrix = comb(n, np.arange(n + 1)) * one_minus_t_powers * t_powers  # (num_samples, n+1)

    # Compute the Bézier curve points
    curve = bernstein_matrix @ control_points  # (num_samples, n+1) @ (n+1, 2) -> (num_samples, 2)
    
    return curve


def visualize_bezier_on_image(image, bezier_curves, widths, save_path, num_samples=100):
    """
    Visualize Bézier curves on an image and save the result to a file.

    Args:
    - image: The background image as a NumPy array of shape (H, W, 3).
    - bezier_curves: Bézier curve control points as a Tensor of shape (N, 4, 2).
    - widths: Widths of the Bézier curves, a Tensor of shape (N,).
    - save_path: Path to save the rendered image.
    - num_samples: Number of points to sample along each Bézier curve.
    """
    # print("image shape: ", image.shape)
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8
    H, W, _ = image_np.shape

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(image_np, extent=[0, W, 0, H])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    # ax.invert_yaxis()

    bezier_curves_np = bezier_curves.cpu().numpy()
    widths_np = widths.cpu().numpy()
    # print("bezier_points: ", bezier_curves_np.shape, bezier_curves_np.max(), bezier_curves_np.min())

    for curve_idx, control_points in enumerate(bezier_curves_np):
        # Scale control points from [0, 1] to [0, W] and [0, H]
        control_points[:, 0] *= W
        control_points[:, 1] *= H

        # Compute Bézier curve points
        bezier_points = bezier_curve(control_points, num_samples=num_samples)

        # Determine the midpoint's pixel color
        midpoint = bezier_points[len(bezier_points) // 2]
        midpoint = np.clip(midpoint, [0, 0], [W - 1, H - 1]).astype(int)
        color = (image_np[midpoint[1], midpoint[0]]) / 255.0  # Normalize color

        # Plot the Bézier curve
        ax.plot(
            bezier_points[:, 0],
            bezier_points[:, 1],
            color=color,
            linewidth=widths_np[curve_idx] * 720,
            solid_capstyle="round",
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    plt.axis("off")
    plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()