#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38
    
# # PDBBind dataset
# python train.py --accelerator gpu \
#     --max_epochs 1 \
#     --precision 16 \
#     --num_layers 2 \
#     --dataset_name PDBBind \
#     --model_name gvp \
#     --input_type complex \
#     --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
#     --num_workers 8 \
#     --residual
    
# PDBBind dataset
python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --protein_num_layers 3 \
    --ligand_num_layers 3 \
    --complex_num_layers 3 \
    --protein_node_h_dim 200 32 \
    --protein_edge_h_dim 64 2 \
    --ligand_node_h_dim 200 32 \
    --ligand_edge_h_dim 64 2 \
    --complex_node_h_dim 200 32 \
    --complex_edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type multistage-physical \
    --model_name gvp-multistage \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10 \