#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

python train.py --accelerator gpu \
    --max_epochs 2 \
    --precision 16 \
    --num_layers 2 \
    --seq_embedding
    
# PDBBind dataset
python train.py --accelerator gpu \
    --max_epochs 2 \
    --precision 16 \
    --num_layers 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8
    