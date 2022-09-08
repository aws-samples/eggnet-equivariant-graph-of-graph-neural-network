#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38
    
python evaluate_casf2016.py \
    --model_name gvp-multistage-energy \
    --input_type multistage-geometric-energy \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/brandry/PDBBind_MSGVPEnergy_geometric/lightning_logs/version_1 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016/ \
    --num_workers 8 \
    --bs 8