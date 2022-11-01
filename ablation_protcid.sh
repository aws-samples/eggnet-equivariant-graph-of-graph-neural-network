#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

n_gpus=4 
num_workers=8
data_dir=/home/ec2-user/SageMaker/efs/data/ProtCID/JaredJanssen_Benchmark_thres_10
residue_featurizer_name=gin-supervised-contextpred-mean # to change this to pretrained GNN residue featurizer
dataset_name=ProtCID
bs=16
lr=1e-4
max_epochs=1000
early_stopping_patience=50
seed=42

node_h_dim=200\ 32
edge_h_dim=64\ 2
num_layers=3

# row2: pretrained GNN	GVP	None
python train.py --accelerator gpu \
    --model_name gvp \
    --devices $n_gpus \
    --num_workers $num_workers \
    --persistent_workers True \
    --precision 16 \
    --dataset_name $dataset_name \
    --input_type complex \
    --residue_featurizer_name $residue_featurizer_name \
    --data_dir $data_dir \
    --data_suffix small \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t10_small_GVP_GIN \
    --random_seed $seed
    
python train.py --accelerator gpu \
    --model_name gvp \
    --devices $n_gpus \
    --num_workers $num_workers \
    --persistent_workers True \
    --precision 16 \
    --dataset_name $dataset_name \
    --input_type complex \
    --residue_featurizer_name $residue_featurizer_name \
    --data_dir $data_dir \
    --data_suffix full \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t10_full_GVP_GIN \
    --random_seed $seed

# row4: pretrained GNN joint training	GVP	None
python train.py --accelerator gpu \
    --model_name hgvp \
    --devices $n_gpus \
    --num_workers 16 \
    --precision 32 \
    --dataset_name $dataset_name \
    --input_type complex \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --data_dir $data_dir \
    --data_suffix small \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t10_small_HGVP_GIN \
    --random_seed $seed


# row6: pretrained GNN	GVP	E_int
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
    --data_dir $data_dir \
    --data_suffix small \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t10_small_GVP_GIN_energy \
    --random_seed $seed

