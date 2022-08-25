#!/bin/bash
# Setting up conda env for this project
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

conda install -y -c rdkit rdkit
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
