#!/bin/bash
# Setting up conda env for this project 
# Note: this requires pytorch_p38 in SageMaker or DLAMI
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
