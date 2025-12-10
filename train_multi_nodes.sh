#!/bin/bash

# please use your dataset path
root=/scratch/xi9/Large-DATASET
dataset=DIV2K_HR
dataset_path="${root}/${dataset}"

if [ ! -d "${dataset_path}" ]; then
    echo "目录不存在: ${dataset_path}"
    exit 1
fi

max_jobs=200
l=6000
counter=0
fre=4
find "${dataset_path}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" \) -print0 | while IFS= read -r -d '' img_path; do
    counter=$((counter+1))
    if [ $((counter % fre)) -ne 0 ]; then
        continue
    fi
    img_name=$(basename "${img_path}")
    sbatch ./scripts/gaussianimage_cholesky/div2k_train.sh "${dataset}" "${img_name}" "${root}" "${counter}"
done
