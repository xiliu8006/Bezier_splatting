# Bézier Splatting for Fast and Differentiable Vector Graphics Rendering 
[[paper](https://arxiv.org/pdf/2503.16424)] [[project page](https://xiliu8006.github.io/Bezier_splatting_project/)]

This is the official implementation of our paper **Bézier Splatting**, an efficient and differentiable vector-graphics representation designed for high-quality image reconstruction and editing. Our method models images using a compact set of Bézier curves, combined with a novel Gaussian-based rasterization that enables fast, stable, and fully differentiable rendering. Thanks to this lightweight vector representation and our adaptive curve optimization strategy, Bézier Splatting achieves high visual fidelity with significantly reduced computation time and memory usage compared to existing differentiable vector graphics methods. Moreover, the explicit curve-based representation makes the output SVGs clean, structured, and truly editable, enabling downstream applications such as scalable dataset generation and fine-grained image manipulation. More qualitative results and details can be found in our paper.

![teaser](./img/teaser.png)

## News
* **2025/9/18**: 🌟 Our paper has been accepted by NeurIPS 2025!

## Overview

![overview](./img/framework.png)

Differentiable vector graphics (VGs) are widely used in image vectorization and vector synthesis, while existing representations are costly to optimize and struggle to achieve high-quality rendering results for high-resolution images. This work introduces a new differentiable VG representation, dubbed **Bézier Splatting**, that enables fast yet high-fidelity VG rasterization. Bézier Splatting samples 2D Gaussians along Bézier curves, which naturally provide positional gradients at object boundaries. Thanks to the efficient splatting-based differentiable rasterizer, Bézier Splatting achieves **30× and 150× faster forward and backward** rasterization for open curves compared to DiffVG. Additionally, we introduce an adaptive pruning and densification strategy that dynamically adjusts the spatial distribution of curves to escape local minima, further improving VG quality. Our representation also supports conversion to standard XML-based SVG format, enhancing interoperability with existing VG tools and pipelines. Experimental results show that Bézier Splatting significantly outperforms existing methods with better visual fidelity and substantial optimization speedup.  
The project page is <https://xiliu8006.github.io/Bezier_splatting_project/>.

## Quick Started

### Cloning the Repository
```shell
# HTTPS
git clone https://github.com/xiliu8006/Bezier_splatting.git

cd Bezier_splatting

#After cloning the repository, please also download the 2D Gaussian rasterization from:

git clone https://github.com/XingtongGe/gsplat/tree/bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d

cd gsplat
pip install .[dev]
```
## Dataset Structure

Our dataset download from [kodak](https://r0k.us/graphics/kodak/) and [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) datasets. The dataset folder is organized as follows.

```bash
├── dataset
│   | kodak 
│     ├── kodim01.png
│     ├── kodim02.png 
│     ├── ...
│   | DIV2K_HR
│     ├── 00001.png
│     ├── 00002.png
│     ├── ...
```
## TO DO
- Release configuration for layer-wise training (a runnable version exists, but further modifications are required).

## Train and Evaluation scripts

To vectorize all images under one folder, we support open and close curves:

```bash
bash train.sh
```

To run on multiple GPUs or multiple nodes:
```bash
bash train_multi_nodes.sh
```

To evaluate performance, run:
```bash
bash get_result.sh
python full_eval.py
```

#### Convert Bezier splatting to standard XML
If you want to get the XML file from our codebase, please save the gaussian model and run the following codes
```
python svg_converter.py
```

## Acknowledgments

We thank [GaussianImage](https://github.com/Xinjie-Q/GaussianImage) for providing the 2D Gaussian rasterization; our code builds upon this excellent foundation.

## Citation

If you find our paper Bézier Splatting useful, please cite:
```
@inproceedings{
liu2025bzier,
title={B\'ezier Splatting for Fast and Differentiable Vector Graphics Rendering},
author={Xi Liu and Chaoyi Zhou and Nanxuan Zhao and Siyu Huang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=bTclOYRfYJ}
}
```

