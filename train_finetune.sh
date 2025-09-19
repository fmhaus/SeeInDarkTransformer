#!/bin/bash

python3 train_model.py \
  --encoder_initial_lr 1e-5 \
  --encoder_weight_decay 0 \
  --bottleneck_initial_lr 1e-4 \
  --bottleneck_weight_decay 0.05 \
  --decoder_initial_lr 3e-5 \
  --decoder_weight_decay 1e-5 \
  --attn_dropout 0.1 \
  --mlp_dropout 0.15 \
  \
  --num_workers 6 \
  --batch_size 3 \
  --effective_batch_size 24 \
  --validation_batch_size 6 \
  --resume_epoch 20 \
  --warmup_epochs 5 \
  --total_epochs 100 \
  --auto_mixed_precision \
  --compile_model \
  --dataset_folder './../dataset/' \
  --preprocess_folder './../preprocess/' \
  --out_folder '/content/drive/MyDrive/SID/out_v1/' \
  --save_checkpoint_frequency 10 