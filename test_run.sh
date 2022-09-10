#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38
    
# PDBBind dataset
python train.py --accelerator gpu \
    --devices -1 \
    --max_epochs 500 \
    --precision 16 \
    --stage1_num_layers 3 \
    --stage1_node_h_dim 200 32 \
    --stage1_edge_h_dim 64 2 \
    --stage2_num_layers 3 \
    --stage2_node_h_dim 200 32 \
    --stage2_edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type multistage-physical \
    --model_name multistage-gvp \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/brandry/PDBBind_MSGVP_physical_energy \
    --use_energy_decoder
    