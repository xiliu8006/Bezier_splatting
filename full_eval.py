import os
import json
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as tf
from pathlib import Path
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.transforms.functional as tf
def readImage(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = tf.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device)

def evaluate(source_root, target_dir):
    """
    Traverse all subdirectories in source_root and search for corresponding
    ground-truth images in target_dir for evaluation.

    Args:
        source_root (str): Root directory containing multiple subdirectories
                           (different methods or experiments).
        target_dir (str): Directory of the ground-truth images (GT).
    """
    if not os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} does not exist!")
        return

    print(f"Using device: {device}")

    for method_dir in sorted(os.listdir(source_root)):
        method_path = os.path.join(source_root, method_dir, 'DIV2K_HR')
        print("method path: ", method_path, source_root)
        if not os.path.isdir(method_path):
            continue

        print(f"Processing method: {method_dir}")

        full_dict = {}
        per_view_dict = {}

        full_dict[method_dir] = {}
        per_view_dict[method_dir] = {}
        source_images = sorted(os.listdir(method_path))

        ssims, psnrs, lpipss = [], [], []

        for image_name in tqdm(source_images, desc=f"Evaluating {method_dir}"):
            source_image_path = os.path.join(method_path, image_name)
            target_image_path = os.path.join(target_dir, image_name)

            if not os.path.exists(target_image_path):
                print(f"Warning: No matching target image for {target_image_path}, skipping.")
                continue

            render = readImage(source_image_path)
            gt = readImage(target_image_path)

            # print("source image path: ", source_image_path)
            # print("render and gt shape", render.shape, gt.shape)
            ssim_val = ssim(render, gt).to(device)
            psnr_val = psnr(render, gt).to(device)
            lpips_val = lpips(render, gt, net_type='vgg').to(device)

            ssims.append(ssim_val)
            psnrs.append(psnr_val)
            lpipss.append(lpips_val)

            per_view_dict[method_dir].setdefault("SSIM", {})[image_name] = ssim_val.item()
            per_view_dict[method_dir].setdefault("PSNR", {})[image_name] = psnr_val.item()
            per_view_dict[method_dir].setdefault("LPIPS", {})[image_name] = lpips_val.item()

        full_dict[method_dir].update({
            "SSIM": torch.stack(ssims).mean().item(),
            "PSNR": torch.stack(psnrs).mean().item(),
            "LPIPS": torch.stack(lpipss).mean().item()
        })

        print(f"  SSIM  ({method_dir}): {full_dict[method_dir]['SSIM']:.7f}")
        print(f"  PSNR  ({method_dir}): {full_dict[method_dir]['PSNR']:.7f}")
        print(f"  LPIPS ({method_dir}): {full_dict[method_dir]['LPIPS']:.7f}")
        print("")

        result_json = os.path.join(source_root, f"{method_dir}_results.json")
        per_view_json = os.path.join(source_root, f"{method_dir}_per_view.json")

        with open(result_json, 'w') as fp:
            json.dump(full_dict[method_dir], fp, indent=True)
        with open(per_view_json, 'w') as fp:
            json.dump(per_view_dict[method_dir], fp, indent=True)

    print("Evaluation completed.")

evaluate("/project/siyuh/common/xiliu/Outputs", "/scratch/xi9/Large-DATASET/DIV2K_HR")