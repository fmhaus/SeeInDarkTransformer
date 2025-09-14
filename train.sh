#!/bin/bash

python3 train_model.py \
  --lr_initial 0.0005 \
  --weight_decay 0.01 \
  --num_workers 4 \
  --batch_size 8 \
  --effective_batch_size 32 \
  --resume_epoch 0 \
  --total_epochs 200 \
  --auto_mixed_precision True \
  --load_optimizer True \
  --encoder_train_factor 0 \
  --dataset_folder './../dataset/' \
  --preprocess_folder './../preprocess/' \
  --out_path '/content/drive/SID/out_v1/' \
  --save_checkpoint_frequency 10 