#!/bin/bash
export OMP_NUM_THREADS=1
torchrun --nproc_per_node 4 \
            train_with_ddp.py \
            --epochs 50 \
            --learning-rate 0.000004 \
            --batch-size 4 \
            --scale 1 \
            --validation 10 \
            --amp \
            --ddp_mode \
            --start_from 48 \
            --load /home/shenlan08/crop_row_segmentation/checkpoints/DDP_checkpoint_epoch48.pth
        

# batch_size refers to the batch_size for a single gpu