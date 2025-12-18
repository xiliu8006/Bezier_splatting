#!/bin/bash

source /etc/profile.d/modules.sh
module add cuda/11.8.0

export PATH=$HOME/miniconda3/bin:$PATH
source activate bs

# please use your dataset path
root='/scratch/xi9/Large-DATASET/'
dataset=DIV2K
dataset_path="${root}/${dataset}"

if [ ! -d "${dataset_path}" ]; then
    echo "dir does not exsit: ${dataset_path}"
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
    # This is for closed curves
    python train_withsvg.py --data_name ${dataset} --image_name ${img_name} -d ${root} \
         --model_name GaussianImage_Cholesky_svg --num_curves 512 --bezier_degree 4 --iterations 10000

    #This is for open curves
    # python train_withsvg.py --data_name ${dataset} --image_name ${img_name} -d ${root} \
    #      --model_name GaussianImage_Cholesky_svg --num_curves 512 --mode unclosed --lr 0.005 --bezier_degree 3 --iterations 15000
done