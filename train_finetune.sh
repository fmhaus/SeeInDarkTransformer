#!/bin/bash

python3 train_model.py \
  --encoder_initial_lr 1e-5 \
  --encoder_weight_decay 0 \
  --bottleneck_initial_lr 1e-4 \
  --bottleneck_weight_decay 0.05 \
  --decoder_initial_lr 3e-5 \
  --decoder_weight_decay 0 \
  \
  --num_workers 6 \
  --batch_size 3 \
  --effective_batch_size 24 \
  --validation_batch_size 6 \
  --resume_epoch 0 \
  --warmup_epochs 10 \
  --total_epochs 200 \
  --auto_mixed_precision True \
  --load_optimizer False \
  --compile_model True \
  --dataset_folder './../dataset/' \
  --preprocess_folder './../preprocess/' \
  --out_path '/content/drive/MyDrive/SID/out_v1/' \
  --save_checkpoint_frequency 10 