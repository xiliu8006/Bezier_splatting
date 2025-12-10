import torch
import os
import numpy as np
import svgwrite
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor


def bezier_splatting_to_svg(control_points, features_dc, svg_path, canvas_width, canvas_height):
    """
    control_points: (N, 8, 2) in [-1, 1]
    features_dc: (N, 3) in [0, 1]
    """
    dwg = svgwrite.Drawing(svg_path, size=(canvas_width, canvas_height), profile='tiny')
    for i in range(control_points.shape[0]):
        pts = control_points[i].cpu().numpy()
        pts = (pts + 1) / 2  # scale from [-1, 1] -> [0, 1]
        pts[:, 0] *= canvas_width
        pts[:, 1] *= canvas_height
        color = features_dc[i].cpu().numpy()
        r, g, b = (color * 255).astype(int)
        hex_color = svgwrite.rgb(r, g, b, mode='rgb')

        n_pts = len(pts)
        assert n_pts % 3 == 0, "Number of control points must be 3n + 1"
        n_segs = (n_pts - 1) // 3
        # A path with 2 cubic Bézier segments (8 control points)
        # assert pts.shape[0] == 6
        # d = (
        #     f"M {pts[0][0]} {pts[0][1]} "
        #     f"C {pts[1][0]} {pts[1][1]}, {pts[2][0]} {pts[2][1]}, {pts[3][0]} {pts[3][1]} "
        #     f"C {pts[4][0]} {pts[4][1]}, {pts[5][0]} {pts[5][1]}, {pts[0][0]} {pts[0][1]} "
        #     f"Z"
        # )
        d = f"M {pts[0][0]} {pts[0][1]} "
        for i in range(n_segs):
            p1 = pts[1 + 3*i]
            p2 = pts[2 + 3*i]
            p3 = pts[3 * i + 3] if (3 * i + 3) < n_pts else pts[0]  # 回到起点
            d += f"C {p1[0]} {p1[1]}, {p2[0]} {p2[1]}, {p3[0]} {p3[1]} "

        d += "Z"  # 闭合路径
        path = dwg.path(d=d, fill=hex_color, stroke='none', fill_opacity=1.0)


        dwg.add(path)
        # break

    dwg.save()


# Load input
image_path = "/home/liuxi/code/DATASET/svg_test/frame_00001.png"
model_path = "/home/liuxi/code/FastDiffVG/checkpoints/DL3DV/GaussianImage_Cholesky_svg_24000_50000/frame_00001/gaussian_model.pth.tar"
ret_dir = "svg"
os.makedirs(ret_dir, exist_ok=True)
device = "cuda:0"
num_curves = 1023

gt_images = image_path_to_tensor(image_path)
H, W = gt_images.shape[2], gt_images.shape[3]

# Initialize model
from gaussianimage_cholesky_svg import GaussianImage_Cholesky
gaussian_model = GaussianImage_Cholesky(
    loss_type="L2", opt_type="adan", num_curves=num_curves, H=H, W=W,
    BLOCK_H=16, BLOCK_W=16, device=device, lr=0.005, mode="closed", quantize=False
).to(device)

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)
model_dict = gaussian_model.state_dict()
# pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
# 过滤掉 key 不存在 或 shape 不匹配的参数
pretrained_dict = {
    k: v for k, v in checkpoint.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
model_dict.update(pretrained_dict)
gaussian_model.load_state_dict(model_dict, strict=False)
# gaussian_model.opacity_activation = torch.nn.Identity()
keep_num = num_curves  # 改成你想保留的数量，比如 1、200、512
print(f"[INFO] Truncating model to first {keep_num} curves.")

gaussian_model._control_points = torch.nn.Parameter(gaussian_model._control_points[:keep_num])
gaussian_model._features_dc = torch.nn.Parameter(gaussian_model._features_dc[:keep_num])
gaussian_model._cholesky = torch.nn.Parameter(gaussian_model._cholesky[:keep_num])
gaussian_model._scaling = torch.nn.Parameter(gaussian_model._scaling[:keep_num])
# gaussian_model._rotation = torch.nn.Parameter(gaussian_model._rotation[:keep_num])
# gaussian_model._xyz = torch.nn.Parameter(gaussian_model._xyz[:keep_num])
gaussian_model._depth = torch.nn.Parameter(gaussian_model._depth[:keep_num])
gaussian_model._opacity = torch.nn.Parameter(gaussian_model._opacity[:keep_num])
renderpkg=gaussian_model()
image = renderpkg['render']
image = image.squeeze(0)

# render_pkg_line = self.gaussian_model.forward_area_boundary()
# image_line = render_pkg_line['render']
# image_line = image_line.squeeze(0)

# print("image shape: ", image.shape)
to_pil = transforms.ToPILImage()
img = to_pil(image)
img.save(f'{ret_dir}/svg.png')

# Convert and export to SVG
control_points = gaussian_model._control_points.detach().cpu()  # (N, 8, 2)
features_dc = torch.sigmoid(gaussian_model._features_dc.detach().cpu())        # (N, 3)

depth = gaussian_model.get_depth.detach()
boxes = gaussian_model.compute_aabb(gaussian_model.xyz.view(gaussian_model.xyz.shape[0], -1, 2))
ratio = gaussian_model.W / gaussian_model.H
# print("ratio: ", ratio)
widths = (boxes[:, 2] - boxes[:, 0]) * ratio
heights = (boxes[:, 3] - boxes[:, 1])
depth = widths * heights
# depth: (512,)
sorted_indices = torch.argsort(depth, descending=True)  # 从大到小排序

# 对 control_points 和 features_dc 同步排序
control_points = control_points[sorted_indices.cpu()]   # shape: (512, 8, 2)
features_dc = features_dc[sorted_indices.cpu()]
svg_output_path = os.path.join(ret_dir, "output.svg")
bezier_splatting_to_svg(control_points, features_dc, svg_output_path, canvas_width=W, canvas_height=H)