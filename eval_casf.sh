#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38
    
python evaluate_casf2016.py \
    --model_name multistage-hgvp \
    --input_type multistage-hetero \
    --residue_featurizer_name MolT5-small-grad \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/brandry/PDBBind_MS-HGVP_hetero_energy/lightning_logs/version_9 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016_processed/ \
    --num_workers 8 \
    --bs 16 \
    --is_hetero \
    --use_energy_decoder