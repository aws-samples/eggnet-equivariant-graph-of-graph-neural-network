#!/bin/bash
# Setting up conda env for this project
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_p38

conda install -y -c rdkit rdkit
pip install -r requirements.txt
