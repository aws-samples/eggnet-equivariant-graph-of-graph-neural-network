#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

# ablation studies for the GVP models on PDBBind regression task
# global variables shared across all runs:
n_gpus=4 
num_workers=8
pdbbind_data=/home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring
residue_featurizer_name=gin-supervised-contextpred-mean # to change this to pretrained GNN residue featurizer
dataset_name=PDBBind
bs=16
lr=1e-4
max_epochs=1000
early_stopping_patience=50
seed=42

node_h_dim=200\ 32
edge_h_dim=64\ 2
num_layers=3

# row2: pretrained GNN	GVP	None
for seed in 43 44; do
    python train.py --accelerator gpu \
        --model_name gvp \
        --devices $n_gpus \
        --num_workers $num_workers \
        --persistent_workers True \
        --precision 16 \
        --dataset_name $dataset_name \
        --input_type complex \
        --residue_featurizer_name $residue_featurizer_name \
        --data_dir $pdbbind_data \
        --bs $bs \
        --lr $lr \
        --max_epochs $max_epochs \
        --early_stopping_patience $early_stopping_patience \
        --residual \
        --node_h_dim $node_h_dim \
        --edge_h_dim $edge_h_dim \
        --num_layers $num_layers \
        --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_GIN \
        --random_seed $seed
done

# row4: pretrained GNN joint training	GVP	None
for seed in 43 44; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --accelerator gpu \
        --model_name hgvp \
        --devices $n_gpus \
        --num_workers $num_workers \
        --precision 32 \
        --dataset_name $dataset_name \
        --input_type complex \
        --residue_featurizer_name $residue_featurizer_name-grad \
        --data_dir $pdbbind_data \
        --bs $bs \
        --lr $lr \
        --max_epochs $max_epochs \
        --early_stopping_patience $early_stopping_patience \
        --residual \
        --node_h_dim $node_h_dim \
        --edge_h_dim $edge_h_dim \
        --num_layers $num_layers \
        --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_HGVP_GIN \
        --random_seed $seed
done

# row6: pretrained GNN	GVP	E_int
for seed in 43 44; do
    python train.py --accelerator gpu \
        --model_name gvp \
        --devices $n_gpus \
        --num_workers $num_workers \
        --persistent_workers True \
        --precision 16 \
        --dataset_name $dataset_name \
        --input_type complex \
        --residue_featurizer_name $residue_featurizer_name \
        --use_energy_decoder \
        --is_hetero \
        --data_dir $pdbbind_data \
        --bs $bs \
        --lr $lr \
        --max_epochs $max_epochs \
        --early_stopping_patience $early_stopping_patience \
        --residual \
        --node_h_dim $node_h_dim \
        --edge_h_dim $edge_h_dim \
        --num_layers $num_layers \
        --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_GIN_energy \
        --random_seed $seed
done

# row8: pretrained GNN joint training	GVP	E_int
n_gpus=8
bs=8
num_workers=8
for seed in 43 44; do
    python train.py --accelerator gpu \
        --model_name hgvp \
        --devices $n_gpus \
        --num_workers $num_workers \
        --precision 32 \
        --dataset_name $dataset_name \
        --input_type complex \
        --residue_featurizer_name $residue_featurizer_name-grad \
        --use_energy_decoder \
        --is_hetero \
        --data_dir $pdbbind_data \
        --bs $bs \
        --lr $lr \
        --max_epochs $max_epochs \
        --early_stopping_patience $early_stopping_patience \
        --residual \
        --node_h_dim $node_h_dim \
        --edge_h_dim $edge_h_dim \
        --num_layers $num_layers \
        --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_HGVP_GIN_energy \
        --random_seed $seed
done

## Evaluation
eval_data_dir=/home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed

# row2: pretrained GNN	GVP	None
python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir $eval_data_dir \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_GIN/lightning_logs/version_0 \
    --residue_featurizer_name $residue_featurizer_name 

# row4: pretrained GNN joint training	GVP	None
python evaluate_casf2016.py --model_name hgvp \
    --num_workers 8 \
    --data_dir $eval_data_dir \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_HGVP_GIN/lightning_logs/version_1 \
    --residue_featurizer_name $residue_featurizer_name-grad


# row6: pretrained GNN	GVP	E_int
python evaluate_casf2016.py --model_name gvp \
    --num_workers 8 \
    --data_dir $eval_data_dir \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_GVP_GIN_energy/lightning_logs/version_0 \
    --residue_featurizer_name $residue_featurizer_name \
    --use_energy_decoder \
    --is_hetero \
    --bs 16

# row8: pretrained GNN joint training	GVP	E_int
python evaluate_casf2016.py --model_name hgvp \
    --num_workers 8 \
    --data_dir $eval_data_dir \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/zichen/PDBBind_HGVP_GIN_energy/lightning_logs/version_3 \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --use_energy_decoder \
    --is_hetero \
    --bs 16