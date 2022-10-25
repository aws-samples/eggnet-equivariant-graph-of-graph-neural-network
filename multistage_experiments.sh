#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

# multistage-physical
for input_type in physical hetero geometric; do 
    CUDA_VISIBLE_DEVICES=2 python train.py --accelerator gpu \
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
        --input_type multistage-$input_type \
        --model_name gvp-multistage \
        --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
        --residual \
        --num_workers 8 \
        --lr 1e-4 \
        --bs 128 \
        --early_stopping_patience 10 \
        --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_MSGVP_$input_type
done

python evaluate_casf2016.py --model_name gvp-multistage \
    --input_type multistage-$input_type \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016 \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_MSGVP_$input_type/lightning_logs/version_2 


CUDA_VISIBLE_DEVICES=1,2 python train.py --accelerator gpu \
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
    --lr 1e-4 \
    --bs 128 \
    --early_stopping_patience 10 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_MSGVP_physical