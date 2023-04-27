# EGGNet: Equivariant Graph-of-Graphs Neural Network

Source code for "[EGGNet, a generalizable geometric deep learning framework for protein complex pose scoring](https://www.biorxiv.org/content/10.1101/2023.03.22.533800v1)"

<img src="figs/GoGs_of_molecules.png">

## Dependencies

All experiments were performed in Python 3.8 with Pytorch (v1.10). 

To install all dependencies run:
```
$ pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
$ pip install -r requirements.txt
```


## Data preparation

PDBbind/CASF-2016 data can be downloaded using [the script](https://github.com/ACE-KAIST/PIGNet/blob/main/data/download_train_data.sh) from the PIGNet repository. 
The included python notebooks can be used as a guide for data prep in order to reproduce results or train on new datasets. This command was used to download DC and MANY data from DeepRank: `rsync -av rsync://data.sbgrid.org/10.15785/SBGRID/843`. Note that the whole download is 500GB. A script for ProtCid-like data can be used for classification tasks (`prep_eggnet_data_protcid_model.py`).  This script will be run twice.  Once for each label.  The input is a directory of PDB structures. 
```
prep_eggnet_data_protcid_model.py --dir many_xtal --label 0 --threshold 12 --skip_filter
prep_eggnet_data_protcid_model.py --dir many_bio --label 1 --threshold 12 --skip_filter --datafile processed/train_full.csv
```

Note that datasets are hard-coded, so if you have a new dataset that is very different than what EggNet was trained on, you will need to modify the code to add a new dataset. 


## Training

Training of EGGNet and competing models for protein complex scoring tasks can be done in `train.py`, which utilizes the [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#). All of the [trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags) in PyTorch Lightning are supported. To see the usage, run: 

```
$ python train.py -h
usage: train.py [-h] [--logger [LOGGER]] [--enable_checkpointing [ENABLE_CHECKPOINTING]] [--default_root_dir DEFAULT_ROOT_DIR] [--gradient_clip_val GRADIENT_CLIP_VAL]
                [--gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES] [--devices DEVICES] [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]]
                [--tpu_cores TPU_CORES] [--ipus IPUS] [--enable_progress_bar [ENABLE_PROGRESS_BAR]] [--overfit_batches OVERFIT_BATCHES] [--track_grad_norm TRACK_GRAD_NORM]
                [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS]
                [--min_epochs MIN_EPOCHS] [--max_steps MAX_STEPS] [--min_steps MIN_STEPS] [--max_time MAX_TIME] [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES]
                [--limit_test_batches LIMIT_TEST_BATCHES] [--limit_predict_batches LIMIT_PREDICT_BATCHES] [--val_check_interval VAL_CHECK_INTERVAL] [--log_every_n_steps LOG_EVERY_N_STEPS]
                [--accelerator ACCELERATOR] [--strategy STRATEGY] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION] [--enable_model_summary [ENABLE_MODEL_SUMMARY]]
                [--weights_save_path WEIGHTS_SAVE_PATH] [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler PROFILER] [--benchmark [BENCHMARK]]
                [--deterministic [DETERMINISTIC]] [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS] [--auto_lr_find [AUTO_LR_FIND]] [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]]
                [--detect_anomaly [DETECT_ANOMALY]] [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--plugins PLUGINS] [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL]
                [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]] [--multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE] [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Choose from gvp, hgvp, multistage-gvp, multistage-hgvp

pl.Trainer:
  --logger [LOGGER]     Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers
  # other pl.Trainer flags...
```

Training scripts for ProtCid and pdbbind can be found in `ablation_protcid.sh` and `ablation_pdbbind.sh`
Example is below for ProtCid-like data joint training with GIN featurizer. 

```
n_gpus=4
num_workers=8

suffix=full

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
crop=12

data_dir=/home/ec2-user/SageMaker/eggnet-equivariant-graph-of-graph-neural-network/crop_${crop}_no_filter
root_dir=/home/ec2-user/SageMaker/eggnet_training_results/crop${crop}

# 3: pretrained GNN joint training	GVP	None
python train.py --accelerator gpu \
    --model_name hgvp \
    --devices $n_gpus \
    --num_workers 16 \
    --precision 32 \
    --dataset_name $dataset_name \
    --input_type complex \
    --residue_featurizer_name $residue_featurizer_name-grad \
    --data_dir $data_dir \
    --data_suffix $suffix \
    --bs $bs \
    --lr $lr \
    --max_epochs $max_epochs \
    --early_stopping_patience $early_stopping_patience \
    --residual \
    --node_h_dim $node_h_dim \
    --edge_h_dim $edge_h_dim \
    --num_layers $num_layers \
    --default_root_dir ${root_dir}/3_ProtCID_t6_small_HGVP_GIN \
    --random_seed $seed
```

## Evaluation
EggNet is dataset-centric, so all inputs will need to be prepped either through the notebook or script. Once prepped, an example evaluation command is below:
```
python evaluate.py --checkpoint_path ../eggnet_training_results/crop12/6_ProtCID_Molt5-small/lightning_logs/version_0 --evaluate_type classification --dataset_name ProtCID --input_type complex --data_suffix full --data_dir /home/ec2-user/SageMaker/eggnet-equivariant-graph-of-graph-neural-network/crop_12_no_filter --residue_featurizer_name MolT5-small-grad --model_name hgvp --num_workers 8 --bs 4 --dataset_alias protcid_test
```

## Citation

Please cite the following preprint:
```
@article {Wang2023.03.22.533800,
	author = {Wang, Zichen and Brand, Ryan and Adolf-Bryfogle, Jared and Grewal, Jasleen and Qi, Yanjun and Combs, Steven A. and Golovach, Nataliya and Alford, Rebecca and Rangwala, Huzefa and Clark, Peter M.},
	title = {EGGNet, a generalizable geometric deep learning framework for protein complex pose scoring},
	elocation-id = {2023.03.22.533800},
	year = {2023},
	doi = {10.1101/2023.03.22.533800},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Computational prediction of molecule-protein interactions has been key for developing new molecules to interact with a target protein for therapeutics development. Literature includes two independent streams of approaches: (1) predicting protein-protein interactions between naturally occurring proteins and (2) predicting the binding affinities between proteins and small molecule ligands (aka drug target interaction, or DTI). Studying the two problems in isolation has limited computational models{\textquoteright} ability to generalize across tasks, both of which ultimately involve non-covalent interactions with a protein target. In this work, we developed Equivariant Graph of Graphs neural Network (EGGNet), a geometric deep learning framework for molecule-protein binding predictions that can handle three types of molecules for interacting with a target protein: (1) small molecules, (2) synthetic peptides and (3) natural proteins. EGGNet leverages a graph of graphs (GoGs) representation constructed from the molecule structures at atomic-resolution and utilizes a multiresolution equivariant graph neural network (GNN) to learn from such representations. In addition, EGGNet gets inspired by biophysics and makes use of both atom- and residue-level interactions, which greatly improve EGGNet{\textquoteright}s ability to rank candidate poses from blind docking. EGGNet achieves competitive performance on both a public proteinsmall molecule binding affinity prediction task (80.2\% top-1 success rate on CASF-2016) and an synthetic protein interface prediction task (88.4\% AUPR). We envision that the proposed geometric deep learning framework can generalize to many other protein interaction prediction problems, such as binding site prediction and molecular docking, helping to accelerate protein engineering and structure-based drug development.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/03/22/2023.03.22.533800},
	eprint = {https://www.biorxiv.org/content/early/2023/03/22/2023.03.22.533800.full.pdf},
	journal = {bioRxiv}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
