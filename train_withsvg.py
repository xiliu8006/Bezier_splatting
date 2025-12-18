import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image

import time
import glob
import matplotlib.pyplot as plt

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        imagesvg_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)
        # self.gt_image_svg = pydiffvg.svg_to_scene(imagesvg_path)
        # print("svg is ", self.gt_image_svg)
        self.num_points = num_points
        self.num_curves = args.num_curves
        self.mode = args.mode
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.num_samples=args.num_samples
        self.bezier_degree = args.bezier_degree

        name = os.path.splitext(args.image_name)[0]
        self.save_path = f'./output/bezier_splatting_{self.mode}_our_{args.num_curves}/{args.data_name}/{name}'
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky_svg":
            from gaussianimage_cholesky_svg import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_curves=self.num_curves, num_samples=args.num_samples, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, mode=self.mode, bezier_degree=self.bezier_degree, quantize=False).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def layerwised_train(self):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        remove_iter = 500
        remove_num = 0
        self.gaussian_model.train()
        start_time = time.time()
        frame_counter = 0
        print("start training the save path is: ", self.save_path)
        current = 1
        M = self.num_curves - 24
        allocations = []
        n = 1
        total = 0

        while total < M:
            alloc = min(n, M - total)
            allocations.append(alloc)
            total += alloc
            n = min(n * 2, 64)
        print("allocations", allocations, M)
        allocations = allocations[1:]
        allocations = allocations[::-1]


        for iter in range(1, self.iterations+1):
            loss, psnr, pred_image = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            if iter % 10 == 0:
                progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                progress_bar.update(10)
            with torch.no_grad():
                if iter % 1000 == 0 and allocations != []:
                    add_path_num = allocations.pop()
                    print("adding path: ", add_path_num)
                    pos_init_method = sparse_coord_init(self.gt_image, pred_image)
                    if iter < 9200:
                        radii = 0.02
                    else:
                        radii = 0.01
                    self.gaussian_model.densify(add_path_num, pos_init_method, self.gt_image, radii)
                self.gaussian_model.optimizer.zero_grad(set_to_none = True)

                if iter == self.iterations:
                    renderpkg=self.gaussian_model()
                    # renderpkg_SR=self.gaussian_model(factor=2)

                    image = renderpkg['render']
                    image = image.squeeze(0)
                    to_pil = transforms.ToPILImage()
                    img = to_pil(image)

                    if iter == self.iterations:
                        img.save(f'{self.save_path}/final.png')
                    frame_counter += 1
                

        # from subprocess import call
        # call(["ffmpeg", "-framerate", "24", "-i",
        #     f'{self.save_path}/svg_%d.png', "-vb", "20M",
        #     f"{self.save_path}/out.mp4"])
        images_path = f'{self.save_path}/svg_*.png'
        for image in glob.glob(images_path):
            # print("remove image is: ", image)
            os.remove(image)


        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        remove_iter = 500
        remove_num = 0
        self.gaussian_model.train()
        start_time = time.time()
        frame_counter = 0
        print("start training the save path is: ", self.save_path)

        for iter in range(1, self.iterations+1):
            if self.mode == 'unclosed':
                loss, psnr, pred_image = self.gaussian_model.train_iter_opencurves(self.gt_image)
            else:
                loss, psnr, pred_image = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
                
                max_iter = 14000 if self.mode == "unclosed" else 9200
                if iter % remove_iter == 0 and 1000 <= iter < max_iter:
                    if (iter // remove_iter) % 2 == 1:  # Odd multiples -> Remove
                        prune_mask = self.gaussian_model.remove_curves_mask()
                        self.gaussian_model.num_curves = prune_mask.sum()
                        remove_num += (~prune_mask).sum()
                        print("keep sum:", self.gaussian_model.num_curves, prune_mask.shape, remove_num)
                        self.gaussian_model.prune_beizer_curves(prune_mask)

                    elif (iter // remove_iter) % 2 == 0 and remove_num > 0:  # Even multiples -> Densify
                        pos_init_method = sparse_coord_init(self.gt_image, pred_image)
                        self.gaussian_model.densify(remove_num, pos_init_method, self.gt_image)
                        print("after densify", self.gaussian_model._control_points.shape, remove_num)
                        remove_num = 0

                self.gaussian_model.optimizer.zero_grad(set_to_none = True)
                # if iter % 1000 == 0:
                if iter == self.iterations:
                    # for factor in [2,4,8,16]:
                    renderpkg=self.gaussian_model()
                    # Render higher resolution
                    # renderpkg_SR=self.gaussian_model(factor=factor, denser_sample=True)
                    image = renderpkg['render']
                    image = image.squeeze(0)
                    to_pil = transforms.ToPILImage()
                    img = to_pil(image)
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    if iter == self.iterations:
                        img.save(f'{self.save_path}/final.png')
                    
                    # render_pkg_line = self.gaussian_model.forward_area_boundary()
                    # image_line = render_pkg_line['render']
                    # image_line = image_line.squeeze(0)
                    # img_line = to_pil(image_line)
                    # img_line.save(f'{self.save_path}/svg_line.png')
                    # frame_counter += 1
                

        # from subprocess import call
        # call(["ffmpeg", "-framerate", "24", "-i",
        #     f'{self.save_path}/svg_%d.png', "-vb", "20M",
        #     f"{self.save_path}/out.mp4"])
        # images_path = f'{self.save_path}/svg_*.png'
        # images_path = f'{self.save_path}/svg_*.png'
        # for image in glob.glob(images_path):
        #     # print("remove image is: ", image)
        #     os.remove(image)

        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='DL3DV', help="Training dataset"
    )
    parser.add_argument(
        "--image_name", type=str, default='0001.jpg', help="Training dataset"
    )
    parser.add_argument(
        "--num_curves", type=int, default=512, help="number of beizer curves"
    )
    parser.add_argument(
        "--bezier_degree", type=int, default=4, help="number of beizer curves"
    )
    parser.add_argument(
        "--num_samples", type=int, default=64, help="number of beizer curves"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--mode", type=str, default="closed", help="model selection: closed, unclosed"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    torch.autograd.set_detect_anomaly(True)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        image_length, start = 24, 0
    elif args.data_name == "DIV2K_valid_LRX2":
        image_length, start = 100, 800
    else:
         image_length, start = 1, 0

    for i in range(start, start+image_length):
        if args.data_name == "kodak":
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'
        elif args.data_name == "DIV2K_HR":
            image_path = Path(args.dataset) / args.data_name / args.image_name
            imagesvg_path =Path(args.dataset) / args.data_name  /  f'final_render.svg'
        elif args.data_name == "Kodak":
            image_path = Path(args.dataset) / args.data_name / args.image_name
            imagesvg_path =Path(args.dataset) / args.data_name  /  f'final_render.svg'
        elif args.data_name == "Clipart2":
            image_path = Path(args.dataset) / args.data_name / args.image_name
            imagesvg_path =Path(args.dataset) / args.data_name  /  f'final_render.svg'
        elif args.data_name == "DIV2K":
            image_path = Path(args.dataset) / args.data_name / args.image_name
            imagesvg_path =Path(args.dataset) / args.data_name  /  f'final_render.svg'
        elif args.data_name == "DL3DV":
            image_path = Path(args.dataset) /  f'0164.png'
            imagesvg_path = Path(args.dataset) /  f'final_render.svg'


        trainer = SimpleTrainer2d(image_path=image_path, imagesvg_path=imagesvg_path,num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        
        # psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.layerwised_train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])
