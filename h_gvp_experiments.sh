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

# GVP with energy
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --num_workers 4 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5/lightning_logs/version_5 \
    --residue_featurizer_name MolT5-small \
    --use_energy_decoder \
    --is_hetero

python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True

## ssGVP with 3 energies
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5/lightning_logs/version_19 \
    --residue_featurizer_name MolT5-small \
    --use_energy_decoder \
    --intra_mol_energy \
    --is_hetero \
    --bs 32

python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=1.0 \
    --loss_der2_ratio=1.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True

## ssGVP with 3 energies (weighted)
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True

python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 3_1
# v14
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 12_1
# v15
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 12_0
# v16
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 3_0
# v17
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=0.0 \
    --loss_der2_ratio=0.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 3_0
# v18
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=0.0 \
    --loss_der2_ratio=0.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 3_1

# v19 load weights from the best 1 energy model
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 3_1 \
    --pretrained_weights /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5/lightning_logs/version_5/checkpoints/epoch=647-step=36936.ckpt

# v20 load weights from the best 1 energy model, 0's on der losses
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=0.0 \
    --loss_der2_ratio=0.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --energy_agg_type 3_1 \
    --pretrained_weights /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5/lightning_logs/version_5/checkpoints/epoch=647-step=36936.ckpt

## ssGVP with 3 energies (weighted) on full PDBBind complexes inputs
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 1000 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PDBBind/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_intact_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True    

python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5/lightning_logs/version_17 \
    --residue_featurizer_name MolT5-small \
    --use_energy_decoder \
    --intra_mol_energy \
    --is_hetero \
    --bs 32 

# Continued training from checkpoint
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name gvp \
    --residue_featurizer_name MolT5-small \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --intra_mol_energy \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 16 \
    --early_stopping_patience 500 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --persistent_workers True \
    --pretrained_weights /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5/lightning_logs/version_5/checkpoints/epoch=337-step=19266.ckpt


python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_3energies_w_MolT5/lightning_logs/version_6/checkpoints/last.ckpt \
    --residue_featurizer_name MolT5-small \
    --use_energy_decoder \
    --intra_mol_energy \
    --is_hetero \
    --bs 32

## ssGVP with energy and end-to-end training
python train.py --accelerator gpu \
    --devices 4 \
    --max_epochs 500 \
    --precision 32 \
    --dataset_name PDBBind \
    --input_type complex \
    --model_name hgvp \
    --residue_featurizer_name MolT5-small-grad \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring \
    --use_energy_decoder \
    --is_hetero \
    --num_workers 8 \
    --lr 1e-4 \
    --bs 4 \
    --early_stopping_patience 50 \
    --loss_der1_ratio=10.0 \
    --loss_der2_ratio=10.0 \
    --min_loss_der2=-20.0 \
    --num_layers 3 \
    --node_h_dim 200 32 \
    --edge_h_dim 64 2 \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5_grad

python evaluate_casf2016.py --model_name hgvp \
    --num_workers 8 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_energy_MolT5_grad/lightning_logs/version_0 \
    --residue_featurizer_name MolT5-small-grad \
    --use_energy_decoder \
    --is_hetero \
    --bs 32

