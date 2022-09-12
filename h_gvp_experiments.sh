#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

# Hierarchical GVP experiments on PDBBind/CASF2016 dataset
# version 29    
python train.py --accelerator gpu \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8

python evaluate.py --model_name gvp \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016/scoring \
    --dataset_alias casf2016_scoring \
    --checkpoint_path /home/ec2-user/SageMaker/ppi-model/lightning_logs/version_29


# version 31
python train.py --accelerator gpu \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10

python evaluate.py --model_name gvp \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016/scoring \
    --dataset_alias casf2016_scoring \
    --checkpoint_path /home/ec2-user/SageMaker/ppi-model/lightning_logs/version_31

python evaluate_casf2016.py --model_name gvp \
    --num_workers 16 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/ppi-model/lightning_logs/version_36

# version 32
python train.py --accelerator gpu \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 3e-3 \
    --bs 128 \
    --early_stopping_patience 10

python evaluate.py --model_name gvp \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016/scoring \
    --dataset_alias casf2016_scoring \
    --checkpoint_path /home/ec2-user/SageMaker/ppi-model/lightning_logs/version_32


python train.py --accelerator gpu \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 4 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10

# a wider GVP:
python train.py --accelerator gpu \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10


# MolT5
python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 0 \
    --lr 1e-4 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MolT5-small \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016 \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5/lightning_logs/version_1 \
    --residue_featurizer_name MolT5-small

for lr in 1e-5 1e-4 1e-3; do
    python train.py --accelerator gpu \
        --devices 1 \
        --max_epochs 500 \
        --precision 16 \
        --num_layers 3 \
        --node_h_dim 200 32 \
        --edge_h_dim 64 2 \
        --dataset_name PDBBind \
        --input_type complex \
        --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
        --residual \
        --num_workers 0 \
        --lr $lr \
        --bs 128 \
        --early_stopping_patience 10 \
        --residue_featurizer_name MolT5-base \
        --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5_base
done


python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 0 \
    --lr 1e-4 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MolT5-large \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5_large

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MolT5_large/lightning_logs/version_1 \
    --residue_featurizer_name MolT5-large

# Morgan fingerprint
python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 0 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name Morgan \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_Morgan

python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 0 \
    --lr 1e-4 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name Morgan \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_Morgan


python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016 \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_Morgan/lightning_logs/version_2 \
    --residue_featurizer_name Morgan

# Repeating MACCS exp on 4-GPU
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MACCS

python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019_processed/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-3 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MACCS

# small batch_size:
python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
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
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MACCS

# on 1-GPU machine
CUDA_VISIBLE_DEVICES=3 python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 128 \
    --early_stopping_patience 501 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MACCS

python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name PDBBind \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --residual \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 128 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS 

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/ppi-model/lightning_logs/version_6 \
    --residue_featurizer_name MACCS

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_MACCS/lightning_logs/version_7 \
    --residue_featurizer_name MACCS
