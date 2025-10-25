# SeeInDarkTransformer

**SeeInDarkTransformer** is a Transformer-based adaptation of "Learning to See in the Dark" in CVPR 2018, by Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.
[Paper](https://arxiv.org/abs/1805.01934)
[Github](https://github.com/cchen156/Learning-to-See-in-the-Dark)

The goal is to enhance low-light image reconstruction on raw sensor data using Transformer-based architectures, extending the "Learning to See in the Dark" approach with global self-attention for better color and detail recovery.

## Project Overview

This project explores in what capacity Transformers - models that excel at capturing long-range relationships, can improve how neural networks reconstruct noisy low-exposure high-resolution images, compared to traditional convolutional approaches.
VisionTransformer models typically struggle to process images in high-resolution because of exploding time complexity and memory usage.  

The solution explored in this project is to use Convolutional Neural Networks to downsample to smaller feature maps and use Transformer architecture on a more managable resolution. Several CNN–Transformer hybrid models are implemented using PyTorch. These models reuse pretrained weights from "Learning to See in the Dark", and are retrained to incorporate new Transformer-based components while using the same dataset.  

The repository includes training and evaluation scripts with extensive configuration options, along with tools for benchmarking, side-by-side comparisons, data preprocessing, and visualization of the training process.

## Visual Comparisons

10006.png
![](./results/10006.png)
10030.png
![](./results/10030.png)
10055.png
![](./results/10055.png)

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

*PSNR and SSIM are standard measures of image reconstruction quality - higher is better.*

**Interpretation:** The Transformer-augmented models achieve slightly better performance metrics with similar inference time and memory usage, suggesting that global context improves reconstruction quality without large computational cost.

## Key Takeaways
- Transformer-augmented models slightly outperform similarly sized CNN baselines on standard performance metrics.
- However, some improvements are not fully captured by these metrics.
- The global context provided by self-attention leads to more coherent color reconstruction, especially on single-colored surfaces and along object boundaries.

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

- Chen Chen et al., *Learning to See in the Dark* (2018).[https://arxiv.org/abs/1805.01934](https://arxiv.org/abs/1805.01934)  
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2021). [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
