#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38
    
python evaluate_casf2016.py \
    --model_name multistage-gvp \
    --input_type multistage-hetero \
    --checkpoint_path /home/ec2-user/SageMaker/efs/model_logs/brandry/PDBBind_MSGVP_hetero_energy/lightning_logs/version_12 \
    --data_dir /home/ec2-user/SageMaker/efs/data/PIGNet/data/casf2016/ \
    --num_workers 8 \
    --bs 16 \
    --use_energy_decoder \
    --is_hetero