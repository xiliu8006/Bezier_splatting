#!/bin/bash
#SBATCH --job-name fastdiffvg
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 64gb
#SBATCH --time 72:00:00
#SBATCH --gpus a100:1
#SBATCH --partition mri2020

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate bs

# please use your dataset path
root=/scratch/xi9/Large-DATASET
dataset=DIV2K
dataset_path="${root}/${dataset}"

if [ ! -d "${dataset_path}" ]; then
    echo "目录不存在: ${dataset_path}"
    exit 1
fi

max_jobs=100
l=6000
counter=0
fre=1
find "${dataset_path}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" \) -print0 | while IFS= read -r -d '' img_path; do
    counter=$((counter+1))
    if [ $((counter % fre)) -ne 0 ]; then
        continue
    fi
    img_name=$(basename "${img_path}")
    # sbatch ./scripts/gaussianimage_cholesky/div2k_train.sh "${dataset}" "${img_name}" "${root}" "${counter}"
    python train_withsvg.py --data_name ${dataset} --image_name ${img_name} -d ${root} \
         --model_name GaussianImage_Cholesky_svg --num_curves 1024 --iterations 12000
done