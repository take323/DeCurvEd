# Deep Curvilinear Editing: Commutative and Nonlinear Image Manipulation for Pretrained Deep Generative Model

This repo contains the source code for our CVPR2023 [paper](https://arxiv.org/abs/2211.14573).

## Abstract

Semantic editing of images is the fundamental goal of computer vision. Although deep learning methods, such as generative adversarial networks (GANs), are capable of producing high-quality images, they often do not have an inherent way of editing generated images semantically. Recent studies have investigated a way of manipulating the latent variable to determine the images to be generated. However, methods that assume linear semantic arithmetic have certain limitations in terms of the quality of image editing, whereas methods that discover nonlinear semantic pathways provide non-commutative editing, which is inconsistent when applied in different orders. This study proposes a novel method called deep curvilinear editing (DeCurvEd) to determine semantic commuting vector fields on the latent space. We theoretically demonstrate that owing to commutativity, the editing of multiple attributes depends only on the quantities and not on the order. Furthermore, we experimentally demonstrate that compared to previous methods, the nonlinear and commutative nature of DeCurvEd provides higher-quality editing.

## Requirements

- Python v3.8.12
- numpy v1.22.1
- PyTorch v1.8.0
- [torchdiffeq v0.2.2](https://github.com/rtqichen/torchdiffeq)

Our codes were made by modifying the codes taken from [WarpedGANSpace](https://github.com/chi0tzp/WarpedGANSpace).

``lib/linear.py`` was made from the code taken from [LinearGANSpace](https://github.com/anvoynov/GANLatentDiscovery).

Pretarined WarpedGANSpace models and pretrained LinearGANSpace models are available from their official repositories.

## Prerequisite pretrained models

Download the prerequisite pretrained models (i.e., GAN generators, face detector, pose estimator, and other attribute detectors), as well as pretrained WarpedGANSpace models (optionally, by passing `-m`), as follows:

```bash
python download_models.py
```

Download pretrained LinearGANSpace models from the original repository and put them on ``experiments/complete/*linear*/models/support_sets.pt``.

Please use the ``experiments/complete/*/args.json`` to evaluate LinearGANSpace and WarpedGANSpace models.

## Training

Commands to train a CurvilinearGANSpace model:

```sh
bash script/train/stylegan2.sh
bash script/train/proggan.sh
bash script/train/biggan.sh
bash script/train/anime.sh
bash script/train/mnist.sh
```

Run `python train.py -h` to show the options in detail.

## Evaluation

Run

```sh
bash script/eval/stylegan2.sh
bash script/eval/proggan.sh
bash script/eval/biggan.sh
bash script/eval/anime.sh
bash script/eval/mnist.sh
```

## Copyrights

Most of our codes were made by modifying the codes taken from other repositories.

- WarpedGANSpace: https://github.com/chi0tzp/WarpedGANSpace.
  - Most files.
- LinearGANSpace: https://github.com/anvoynov/GANLatentDiscovery.
  - `models/linear.py`
- StyleFlow: https://github.com/RameenAbdal/StyleFlow.
  - `lib/diffeq_layers.py`
  - `lib/cnf.py`
  - `lib/ffjord.py`
  - `lib/normalization.py`
  - `lib/odefunc.py`

## Reference

```bibtex
@InProceedings{Aoshima_2023_CVPR,
    author    = {Aoshima, Takehiro and Matsubara, Takashi},
    title     = {Deep Curvilinear Editing: Commutative and Nonlinear Image Manipulation for Pretrained Deep Generative Model},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5957-5967}
}
```
