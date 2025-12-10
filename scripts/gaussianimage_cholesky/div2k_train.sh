#!/bin/bash
#SBATCH --job-name fastdiffvg
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 64gb
#SBATCH --time 4:00:00
#SBATCH --gpus a100:1
#SBATCH --partition mri2020

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate svgrender

# CUDA_VISIBLE_DEVICES=0 python train_withsvg.py --data_name $1 --image_name $2 -d $3 \
#      --model_name GaussianImage_Cholesky_svg --num_curves 128 --iterations 10000
CUDA_VISIBLE_DEVICES=0 python train_withsvg.py --data_name $1 --image_name $2 -d $3 \
     --model_name GaussianImage_Cholesky_svg --num_curves 256  --iterations 10000
CUDA_VISIBLE_DEVICES=0 python train_withsvg.py --data_name $1 --image_name $2 -d $3 \
     --model_name GaussianImage_Cholesky_svg --num_curves 512 --iterations 10000
CUDA_VISIBLE_DEVICES=0 python train_withsvg.py --data_name $1 --image_name $2 -d $3 \
     --model_name GaussianImage_Cholesky_svg --num_curves 1024 --iterations 10000
