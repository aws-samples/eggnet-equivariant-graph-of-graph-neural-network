#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Run this on Amazon DLAMI or SageMaker
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

# global variables shared across all tests:
n_gpus=4 
pdbbind_data=/home/ec2-user/SageMaker/efs/data/PIGNet/data/pdbbind_v2019/scoring
residue_featurizer_name=gin-supervised-contextpred-mean
# residue_featurizer_name=MACCS

# row2: pretrained GNN	GVP	None
python train.py --accelerator gpu \
    --model_name gvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --residual \
    --residue_featurizer_name $residue_featurizer_name \
    --data_dir $pdbbind_data

# row3: pretrained GNN	MS-GVP	None 
python train.py --accelerator gpu \
    --model_name multistage-gvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type multistage-hetero \
    --residual \
    --residue_featurizer_name $residue_featurizer_name \
    --data_dir $pdbbind_data

# row4: pretrained GNN joint training	GVP	None
python train.py --accelerator gpu \
    --model_name hgvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 32 \
    --dataset_name PDBBind \
    --input_type complex \
    --residual \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --data_dir $pdbbind_data

# row5: pretrained GNN joint training	MS-GVP	None
python train.py --accelerator gpu \
    --model_name multistage-hgvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 32 \
    --dataset_name PDBBind \
    --input_type multistage-hetero \
    --is_hetero \
    --residual \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --data_dir $pdbbind_data

# row6: pretrained GNN	GVP	E_int
python train.py --accelerator gpu \
    --model_name gvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type complex \
    --residual \
    --residue_featurizer_name $residue_featurizer_name \
    --use_energy_decoder \
    --is_hetero \
    --data_dir $pdbbind_data

# row7: pretrained GNN	MS-GVP	E_int
python train.py --accelerator gpu \
    --model_name multistage-gvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 16 \
    --dataset_name PDBBind \
    --input_type multistage-hetero \
    --residual \
    --residue_featurizer_name $residue_featurizer_name \
    --use_energy_decoder \
    --is_hetero \
    --data_dir $pdbbind_data

# row8: pretrained GNN joint training	GVP	E_int
python train.py --accelerator gpu \
    --model_name hgvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 32 \
    --dataset_name PDBBind \
    --input_type complex \
    --residual \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --use_energy_decoder \
    --is_hetero \
    --data_dir $pdbbind_data

# row9: pretrained GNN joint training	MS-GVP	E_int
python train.py --accelerator gpu \
    --model_name multistage-hgvp \
    --devices $n_gpus \
    --fast_dev_run $n_gpus \
    --precision 32 \
    --dataset_name PDBBind \
    --input_type multistage-hetero \
    --residual \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --use_energy_decoder \
    --is_hetero \
    --data_dir $pdbbind_data \
    --bs 8
