[![Contributors][contributors-shield]][contributors-url]
[![Contributor1](https://github.com/fafraob.png?size=40)](https://github.com/fafraob)
[![Contributor2](https://github.com/MichaelSchwingshackl.png?size=40)](https://github.com/MichaelSchwingshackl)&nbsp;&nbsp;&nbsp;
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]




# [WIP] PI3DETR: Parametric Instance Detection of 3D Point Cloud Edges with a Geometry-Aware 3DETR



[contributors-shield]: https://img.shields.io/github/contributors/fafraob/pi3detr.svg?style=for-the-badge
[contributors-url]: https://github.com/fafraob/pi3detr/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/fafraob/pi3detr.svg?style=for-the-badge
[forks-url]: https://github.com/fafraob/pi3detr/network/members
[stars-shield]: https://img.shields.io/github/stars/fafraob/pi3detr.svg?style=for-the-badge
[stars-url]: https://github.com/fafraob/pi3detr/stargazers
[issues-shield]: https://img.shields.io/github/issues/fafraob/pi3detr.svg?style=for-the-badge
[issues-url]: https://github.com/fafraob/pi3detr/issues
[license-shield]: https://img.shields.io/github/license/fafraob/pi3detr.svg?style=for-the-badge
[license-url]: https://github.com/fafraob/pi3detr/blob/master/LICENSE




We present **PI3DETR**, an end-to-end framework that directly predicts 3D parametric curve instances from raw point clouds, avoiding the intermediate representations and multi-stage processing common in prior work.

[[`website`](https://fafraob.github.io/pi3detr/)] [[`arXiv`](https://arxiv.org/pdf/2509.03262)] [[`bibtex`](#citation)]

![](assets/pi3detr_logo_anim.gif)
Extending [3DETR](https://github.com/facebookresearch/3detr.git), our model introduces a **geometry-aware matching strategy** and specialized loss functions that enable unified detection of differently parameterized curve types, including cubic Bézier curves, line segments, circles, and arcs, in a single forward pass. Optional post-processing steps further refine predictions without adding complexity. This streamlined design improves robustness to noise and varying sampling densities, addressing critical challenges in real world LiDAR and 3D sensing scenarios. PI3DETR sets a new state-of-the-art on the ABC dataset and generalizes effectively to real sensor data, offering a simple yet powerful solution for 3D edge and curve estimation.

![](assets/architecture.png)

## Installation
Our code is tested with PyTorch 2.5.1, CUDA 12.1 and Python 3.11.10. It may and probably will work with other versions too.

You will simply need to install the required dependencies using pip in your preferred python environment (e.g. venv or conda), e.g.:

```bash
pip install -r requirements.txt
```
We also provide a [Dockerfile](Dockerfile) if a containerized environment is preferred.

## Running PI3DETR
Pre-trained checkpoint and dataset is available on [Zenodo](https://zenodo.org/records/16918246). Put the checkpoint under `checkpoints/` to make the commands work without changing the parameters. For the evaluations, put the downloaded dataset in the working directory.

### Inference
To run and visualize the demo samples, use
```bash
python predict_pi3detr.py \
    --config configs/pi3detr.yaml \
    --checkpoint checkpoints/checkpoint.ckpt \
    --path demo_samples \
    --sample_mode all
```
Given the checkpoint (`--checkpoint`) and the config file (`--config`), the script runs inference on the input file or folder specified by `--path`. Supported file formats include .ply, .obj, .pt, and .xyz.

When dealing with huge point clouds, you may want to adjust the sampling parameters to reduce memory usage and improve inference speed. For example, you can use the `--samples` argument to limit the number of points processed, and the `--reduction` argument to downsample the point cloud before applying the main sampling strategy.
```bash
python predict_pi3detr.py \
    --config configs/pi3detr.yaml \
    --checkpoint checkpoints/checkpoint.ckpt \
    --path path_to_your_huge_pc \
    --samples 32768 \
    --sample_mode fps \
    --reduction 100000
```
In this case, the point cloud will be reduced to 100,000 points with random sampling before applying the farthest point sampling strategy to obtain the final 32,768 points.

If you want to change the number of queries used during inference, which is possible since we do use non-parametric queries, you can adjust `num_preds` in the config file.

### Train
To train the model, use the following command:
```bash
python train.py --config configs/pi3detr.yaml
```
The `--config` file specifies all the hyperparameters and settings for training the model. You can adjust it to match your dataset and experiment requirements. For the dataset directories `data_root`, `data_val_root` and `data_test_root` if you set such a path as `/abc_dataset/train`, the code will look for the data in `/abc_dataset/train/processed` following the [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) convention. To save time during training, we assume the data to be already preprocessed.


## Examples
![](assets/big_comparison.jpg)


## Evaluation
We compare our method with [NerVE](https://github.com/uhzoaix/NerVE), which shows strong performance, particularly on point clouds containing many points along object edges. In contrast, the data used in this project more closely resembles real 3D scans from LiDAR or similar sensors. This is because point clouds are obtained using a surface sampling approach on meshes from the [ABC Dataset](https://deep-geometry.github.io/abc-dataset/).


The commands to reproduce the results shown in the following tables are provided below.  
Make sure to update the `data_test_root` field in the corresponding config file to match your dataset path.

```bash
python3 evaluate_pi3detr.py --config configs/pi3detr_k256.yaml --checkpoint checkpoints/checkpoint.ckpt -v
```
| Metric        | NerVE CAD                          | NerVE PWL                          | **Ours**                           |
|---------------|------------------------------------|------------------------------------|------------------------------------|
| **CD ↓**      | 0.0401 (± 0.20)                    | 0.0046 (± 0.02)                    | **0.0024 (± 0.02)**                |
| **HD ↓**      | 0.2478 (± 0.35)                    | 0.1534 (± 0.17)                    | **0.0635 (± 0.07)**                |
| **mAP ↑**     | --                                 | --                                 | **0.8090**                         |


To reproduce the subsampling experiments, add `--samples` followed by the desired number of points.
```bash
python3 evaluate_pi3detr.py --config configs/pi3detr_k256.yaml --checkpoint checkpoints/checkpoint.ckpt -v --samples 4096
```

| **N**    | NerVE CAD             | NerVE PWL             | **Ours**            |
|----------|-----------------------|-----------------------|---------------------|
|          | *Chamfer Distance (CD ↓)* |                       |                     |
| 32,768   | 0.0401 (± 0.20)       | 0.0046 (± 0.02)       | **0.0024 (± 0.02)** |
| 16,384   | 0.1134 (± 0.43)       | 0.0061 (± 0.02)       | **0.0025 (± 0.02)** |
| 8,192    | 0.2882 (± 0.60)       | 0.0167 (± 0.04)       | **0.0027 (± 0.02)** |
| 4,096    | 0.4562 (± 0.68)       | 0.0984 (± 0.27)       | **0.0050 (± 0.03)** |
|          | *Hausdorff Distance (HD ↓)* |                   |                     |
| 32,768   | 0.2477 (± 0.35)       | 0.1534 (± 0.17)       | **0.0635 (± 0.07)** |
| 16,384   | 0.3961 (± 0.49)       | 0.2008 (± 0.20)       | **0.0634 (± 0.07)** |
| 8,192    | 0.6436 (± 0.66)       | 0.2987 (± 0.25)       | **0.0680 (± 0.08)** |
| 4,096    | 0.8665 (± 0.74)       | 0.4918 (± 0.42)       | **0.0857 (± 0.10)** |


For noise robustness evaluation, use the following command with `--noise`, where you specify the noise level.
```bash
python3 evaluate_pi3detr.py --config configs/pi3detr_k256.yaml --checkpoint checkpoints/checkpoint.ckpt -v --noise 2e2
```

| **Noise**       | NerVE CAD           | NerVE PWL           | **Ours**            |
|-----------------|---------------------|---------------------|---------------------|
|                 | *Chamfer Distance (CD ↓)* |                 |                     |
| η = s/1e³       | 0.0311 (± 0.21)     | **0.0061 (± 0.02)** | 0.0113 (± 0.06)     |
| η = s/5e²       | 0.0194 (± 0.13)     | **0.0121 (± 0.04)** | 0.0129 (± 0.06)     |
| η = s/2e²       | 0.0164 (± 0.04)     | 0.0211 (± 0.05)     | **0.0134 (± 0.06)** |
|                 | *Hausdorff Distance (HD ↓)* |              |                     |
| η = s/1e³       | 0.2306 (± 0.30)     | 0.2530 (± 0.22)     | **0.1471 (± 0.11)** |
| η = s/5e²       | 0.2743 (± 0.25)     | 0.3581 (± 0.23)     | **0.1684 (± 0.12)** |
| η = s/2e²       | 0.3086 (± 0.20)     | 0.3874 (± 0.23)     | **0.1946 (± 0.12)** |

## Citation

```bibtex
@misc{oberweger2025pi3detrparametricinstancedetection,
      title={PI3DETR: Parametric Instance Detection of 3D Point Cloud Edges with a Geometry-Aware 3DETR}, 
      author={Fabio F. Oberweger and Michael Schwingshackl and Vanessa Staderini},
      year={2025},
      eprint={2509.03262},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.03262}, 
}

```
