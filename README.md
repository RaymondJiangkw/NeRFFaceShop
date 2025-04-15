## NeRFFaceShop: Learning a Photo-realistic 3D-aware Generative Model of Animatable and Relightable Heads from Large-scale In-the-wild Videos <br><sub>Official PyTorch implementation of the TVCG 2025 paper</sub>

![Teaser image](./docs/teaser.png)

**NeRFFaceShop: Learning a Photo-realistic 3D-aware Generative Model of Animatable and Relightable Heads from Large-scale In-the-wild Videos**<br>
Kaiwen Jiang, Feng-Lin Liu, Shu-Yu Chen, Pengfei Wan, Yuan Zhang, Yu-Kun Lai, Hongbo Fu, and Lin Gao<br>

[**Paper**](https://doi.ieeecomputersociety.org/10.1109/TVCG.2025.3560869)

Abstract: *Animatable and relightable 3D facial generation has fundamental applications in computer vision and graphics.
Although animation and relighting are highly correlated, previous methods usually address them separately. Effectively combining animation methods and relighting methods is nontrivial. In terms of explicit shading models, animatable methods cannot be easily extended to achieve realistic relighting results, such as shadow effects, due to prohibitive computational training costs. Regarding implicit lighting representations, current animatable methods cannot be incorporated due to their inharmonious animation representations, i.e., deforming spatial points. This paper, armed
with a lightweight but effective lighting representation, presents a compatible animation representation to achieve a disentangled generative model of 3D animatable and relightable heads. Our represented animation allows for updating and control of realistic lighting effects. Due to the disentangled nature of our representations, we learn the animation and relighting from large-scale, in-the-wild videos instead of relying on a morphable model. We show that our method can synthesize geometrically consistent and detailed motion along with the disentangled control of lighting conditions. We further show that our method is still compatible with morphable models for driving generated avatars. Our method can also be extended to domains without video data by domain transfer to achieve a broader range of animatable
and relightable head synthesis. We will release the code for reproducibility and facilitating future research.*

## Requirements

* We have done all training, testing and development on the Linux platform. We recommend 1 high-end NVIDIA GPU for testing, and 4+ high-end NVIDIA GPUs for training.
* 64-bit Python 3.9 and PyTorch 2.5.1 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 12.4 or later. (Why is a separate CUDA toolkit installation required?  We use the custom CUDA extensions from the StyleGAN3 repo. Please see [Troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
```bash
$ conda env create -f environment.yml
$ conda activate nerffaceshop
$ python -m ipykernel install --user --name=nerffaceshop
```
* We also require Git LFS to be installed for handling large files.

## Getting Started

- Please go to this [link](https://drive.google.com/drive/folders/1jdTj4w02gl18nn7ttZy-DCVVXkvkSogr?usp=sharing) to download content and put it under `./data`.
- Please go to this [link](https://drive.google.com/drive/folders/1f_U42kqR5oCRsHjs5FT-wIfbFf3fCC3G?usp=sharing) to download content and put it under `./external_dependencies/data`

## Generating Media

We provide a notebook `./nerffaceshop/demo.ipynb` for demonstrating how to generate a sample while controlling the lighting condition and expression. We also provide the demonstration of using our adapted model to domains other than human faces.

## Preparing Datasets
We first train the rolled-out version of [EG3D](https://github.com/NVlabs/eg3d) on the FFHQ dataset, please refer to [this repository](https://github.com/NVlabs/eg3d) for its downloading and preprocessing. While for testing, we also require the SH labelling, please refer to [this repository](https://github.com/IGLICT/NeRFFaceLighting) for downloading the supplemented label file.

We train our network on the CelebV-Text dataset and VFHQ dataset, which is in total too large for us to provide the processed version. To reproduce the processed dataset, please refer to [this repository](https://github.com/celebv-text/CelebV-Text) and [this repository](https://github.com/jinwonkim93/vfhq_processing) for downloading. The resulting structure is expected to be an input folder containing numerious video files. After then, please execute the following processing scripts one by one:
```bash
$ cd dataset_preprocessing
$ python 1_process_video.py --in_dir <path to input folder> --out_dir <path to output folder>
$ python 2_detect_flame_coeffs.py --out_dir <path to output folder>
$ python 3_filter_subtle.py --out_dir <path to output folder>
$ python 4_group_for_sampling.py --out_dir <path to output folder>
$ python 5_label.py --out_dir <path to output folder>
```

## Training
Please remember to **update some paths** before training.

You can use the following command with 4+ high-end NVIDIA GPUs to fine-tune the rolled-out version of EG3D (while we also provide our pre-trained model at [here](#getting-started), named as `ffhq512-64-rolled.pkl`):
```bash
$ cd eg3d-rolled
$ ./start.sh
```

You can also use the following command with 4+ high-end NVIDIA GPUs to train our relightable and animatable model on the processed video dataset (while we also provide our pre-trained model at [here](#getting-started), named as `vfhq-celebv-text-64.pkl`):
```bash
$ cd nerffaceshop
$ ./start.sh
```

The training of the MLP to map the coefficients of 3DMM into the motion latent codes of our model requires 1 high-end NVIDIA GPU is written in a jupyter notebook file (`./nerffaceshop/encoder.ipynb`) (while we also provide our pre-trained model at [here](#getting-started), named as `encoder.pth`).

## Citation
```
@article{nerffaceshop,
  author  = {Jiang, Kaiwen and Liu, Feng-Lin and Chen, Shu-Yu and Wan, Pengfei and Zhang, Yuan and Lai, Yu-Kun and Fu, Hongbo and Gao, Lin},
  title   = {NeRFFaceShop: Learning a Photo-realistic 3D-aware Generative Model of Animatable and Relightable Heads from Large-scale In-the-wild Videos},
  year    = {2025},
  journal = {IEEE Transactions on Visualization and Computer Graphics}
}
```

## Acknowledgements
This repository relies on the [EG3D](https://github.com/NVlabs/eg3d), [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), [FaceBoxes](https://github.com/sfzhang15/FaceBoxes), [3DDFA](https://github.com/cleardusk/3DDFA_V2), [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads), [DECA](https://github.com/yfeng95/DECA), and [DPR](https://github.com/zhhoper/DPR).
Please notice that our certain reliant packages (i.e., FLAME, BFM) for preprocessing enforce specific regulations. By using our repository, you agree to those regulations.