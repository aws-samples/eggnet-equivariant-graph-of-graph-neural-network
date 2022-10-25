#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

###########
## Propedia
###########
python train.py --accelerator gpu \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --residual \
    --num_workers 8 \
    --bs 32 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_GVP_MACCS


python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 2 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --residual \
    --num_workers 16 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_GVP_MACCS 


python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --data_suffix small \
    --residual \
    --num_workers 4 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_small_GVP_MACCS \
    --persistent_workers True


# selfdock: refine the positive complexes
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --data_suffix small_selfdock \
    --residual \
    --num_workers 4 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_small_GVP_MACCS \
    --persistent_workers True

# selfdock + noise
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --data_suffix small_selfdock \
    --residual \
    --num_workers 4 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_small_GVP_MACCS \
    --add_noise 0.02

# crystal + noise
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --data_suffix small \
    --residual \
    --num_workers 4 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_small_GVP_MACCS \
    --add_noise 0.02

# less persistent workers to save CPU RAM
# -> still may run out of CPU RAM
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/Propedia \
    --residual \
    --num_workers 8 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/Propedia_GVP_MACCS \
    --persistent_workers True

###########
## PDBBind binary classification
###########
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
    --num_workers 4 \
    --bs 32 \
    --lr 1e-4 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_bin_GVP_MACCS \
    --persistent_workers True \
    --binary_cutoff 6.7

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
    --num_workers 4 \
    --bs 32 \
    --lr 1e-4 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MolT5-small \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_bin_GVP_MolT5_small \
    --persistent_workers True \
    --binary_cutoff 6.7

###########
## ProtCID
###########
python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/ProtCID/JaredJanssen_Benchmark \
    --data_suffix small \
    --residual \
    --num_workers 8 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MACCS \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_small_GVP_MACCS \
    --persistent_workers True

python train.py --accelerator gpu \
    --devices 1 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name Propedia \
    --input_type complex \
    --data_dir /home/ec2-user/SageMaker/efs/data/ProtCID/JaredJanssen_Benchmark \
    --data_suffix small \
    --residual \
    --num_workers 4 \
    --bs 16 \
    --lr 1e-3 \
    --early_stopping_patience 10 \
    --residue_featurizer_name MolT5-small \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_small_GVP_MolT5_small \
    --persistent_workers True