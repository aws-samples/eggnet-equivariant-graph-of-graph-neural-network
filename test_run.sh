#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

python train.py --accelerator gpu \
    --max_epochs 2 \
    --precision 16 \
    --num_layers 2
    