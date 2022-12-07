#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

n_gpus=4 
num_workers=8
# data_dir=/home/ec2-user/SageMaker/efs/data/ProtCID/JaredJanssen_Benchmark_thres_10
data_dir=/home/ec2-user/SageMaker/efs/data/ProtCID/JaredJanssen_Benchmark_thres_6
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
    --data_suffix small_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t6_small_GVP_GIN \
    --random_seed $seed

# row3: pretrained GNN  MS-GVP None
python train.py --accelerator gpu \
    --model_name multistage-gvp \
    --devices $n_gpus \
    --num_workers $num_workers \
    --persistent_workers True \
    --precision 16 \
    --dataset_name $dataset_name \
    --input_type multistage-complex \
    --residue_featurizer_name $residue_featurizer_name \
    --data_dir $data_dir \
    --data_suffix small_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --stage1_node_h_dim $node_h_dim \
    --stage1_edge_h_dim $edge_h_dim \
    --stage1_num_layers $num_layers \
    --stage2_node_h_dim $node_h_dim \
    --stage2_edge_h_dim $edge_h_dim \
    --stage2_num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/brandry/ProtCID_t6_small_MS-GVP_GIN \
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
    --data_suffix small_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t6_small_HGVP_GIN \
    --random_seed $seed

python train.py --accelerator gpu \
    --model_name hgvp \
    --devices $n_gpus \
    --num_workers 16 \
    --precision 32 \
    --dataset_name $dataset_name \
    --input_type complex \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --data_dir $data_dir \
    --data_suffix full_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t6_full_HGVP_GIN \
    --random_seed $seed

# row5: pretrained GNN  joint training MS-GVP None
python train.py --accelerator gpu \
    --model_name multistage-hgvp \
    --devices $n_gpus \
    --num_workers $num_workers \
    --precision 32 \
    --dataset_name $dataset_name \
    --input_type multistage-complex \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --data_dir $data_dir \
    --data_suffix small_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --stage1_node_h_dim $node_h_dim \
    --stage1_edge_h_dim $edge_h_dim \
    --stage1_num_layers $num_layers \
    --stage2_node_h_dim $node_h_dim \
    --stage2_edge_h_dim $edge_h_dim \
    --stage2_num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/brandry/ProtCID_t6_small_MS-HGVP_GIN \
    --random_seed $seed

# row6: pretrained GNN	GVP	E_int
bs=4
lr=1e-4
n_gpus=4
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
    --data_suffix small_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t6_small_GVP_GIN_energy \
    --random_seed $seed \
    --loss_der1_ratio 0 \
    --loss_der2_ratio 0

# row6: with final energy bias
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
    --data_suffix small_filt1e5 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t6_small_GVP_GIN_energy \
    --random_seed $seed \
    --loss_der1_ratio 0 \
    --loss_der2_ratio 0 \
    --final_energy_bias

# row6: smaller network
bs=4
lr=1e-3
n_gpus=8
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
    --data_suffix small_filt1e6 \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --default_root_dir /home/ec2-user/SageMaker/efs/model_logs/zichen/ProtCID_t10_small_GVP_GIN_energy \
    --random_seed $seed \
    --loss_der1_ratio 0 \
    --loss_der2_ratio 0

