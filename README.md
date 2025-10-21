# SeeInDarkTransformer

SeeInDarkTransformer is a Transformer-based adaptation of "Learning to See in the Dark" in CVPR 2018, by Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.
[Paper](https://arxiv.org/abs/1805.01934)
[Github](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Project Overview

This project explores the potential of self-attention mechanisms to capture spatial dependencies for high-quality image reconstruction.  

It implements several CNN–Transformer hybrid models using PyTorch. These models leverage CNN layers and pretrained weights from Learning to See in the Dark, and are retrained to incorporate Transformer-based components while using the same dataset.  

The repository includes training and evaluation scripts with extensive configuration options, along with tools for benchmarking, side-by-side comparisons, data preprocessing, and visualization of the training process.

## Quantitative Comparison

| Feature | Original (CNN-based) | Transformer-adaptation (2B) | Transformer-adaptation (4B) |
|---------|----------------------|-----------------------------|-----------------------------|
| Architecture | Convolutional UNet (5 Encoder + 4 Decoder layers) | 4 Conv-Encoder layers, 2 Transformer-Encoder blocks, 4 Conv-Decoder layers | 4 Conv-Encoder layers, 2 Transformer-Encoder blocks, 4 Conv-Decoder layers |
| Receptive field | Encoder: 248px  (5.83%) Decoder: 296px (6.95%) | Whole image | Whole image |
| Parameters | 7760268 | 6324620 | 7969932 |
| Inference time (L4 GPU, single batch, averaged) | 0.131s | 0.132s | 0.137s |
| Inference VRAM (L4 GPU, single batch, averaged) | 2.44GB | 2.43GB | 2.44GB |
| PSNR (average) | 28.54 dB | 29.60 dB | 29.64 dB |
| SSIM (average) | 0.80 | 0.81 | 0.81 | 

## Visual Comparisons

10006.png
![](./results/10006.png)
10030.png
![](./results/10030.png)
10055.png
![](./results/10055.png)

## Setup

- Download the dataset using `download_dataset_sony.sh` or from the dataset creator [here](https://github.com/cchen156/Learning-to-See-in-the-Dark) and unzip manually.
- Install required libraries: `pip install -r requirements.txt`

## Training

```bash
python train_model.py --model 'sid_bottleneck_transformer_2b' --dataset_folder './dataset/'
```

Check out [config.py](./config.py) and [train_model.py](./train_model.py) for more config options.  
Check out [l4.ini](./config/l4.ini) on how to configure device options.

## Evaluation 

```bash
python test_model.py --model 'sid_bottleneck_transformer_2b' --dataset_folder './dataset' --save_images
```

## References

Chen Chen et al., *Learning to See in the Dark* (2018).[https://arxiv.org/abs/1805.01934](https://arxiv.org/abs/1805.01934)  
Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2021). [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
