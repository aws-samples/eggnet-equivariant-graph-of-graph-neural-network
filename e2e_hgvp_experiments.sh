#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

python train.py --model_name hgvp \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 32 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 8 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MolT5-small-grad \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5_grad

python train.py --model_name hgvp \
    --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 32 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 4 \
    --early_stopping_patience 200 \
    --residue_featurizer_name MolT5-small-grad \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5_grad

python evaluate_casf2016.py --model_name hgvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5_grad/lightning_logs/version_23 \
    --residue_featurizer_name MolT5-small-grad


## PDBBind bin-clf
python train.py --model_name hgvp \
    --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 32 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-5 \
    --bs 4 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MolT5-small-grad \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_bin_GVP_MolT5_grad \
    --binary_cutoff 6.7
