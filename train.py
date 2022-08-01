import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import json
import pickle
from pprint import pprint
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import dgl

import torchmetrics

# custom imports
# from ppi.modules import GATModel, GVPModel
from ppi.model import LitGVPModel
from ppi.data import prepare_pepbdb_data_list, PepBDBComplexDataset
from ppi.data_utils import NaturalComplexFeaturizer

# mapping model names to constructors
MODEL_CONSTRUCTORS = {
    "gvp": LitGVPModel,
    # "gat": GATModel,
}


def init_model(datum=None, model_name="gvp", num_outputs=1, **kwargs):
    if "gvp" in model_name:
        node_in_dim = (
            datum.ndata["node_s"].shape[1],
            datum.ndata["node_v"].shape[1],
        )
        kwargs["node_h_dim"] = tuple(kwargs["node_h_dim"])
        edge_in_dim = (
            datum.edata["edge_s"].shape[1],
            datum.edata["edge_v"].shape[1],
        )
        kwargs["edge_h_dim"] = tuple(kwargs["edge_h_dim"])
        print("node_h_dim:", kwargs["node_h_dim"])
        print("edge_h_dim:", kwargs["edge_h_dim"])

        model = MODEL_CONSTRUCTORS[model_name](
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            num_outputs=num_outputs,
            **kwargs
        )
    else:
        model = MODEL_CONSTRUCTORS[model_name](
            in_feats=datum.ndata["node_s"].shape[1],
            num_outputs=num_outputs,
            **kwargs
        )

    return model


def get_datasets(name="PepBDB", input_type="complex"):
    if name == "PepBDB":
        # load parsed PepBDB structures
        train_structs = pickle.load(
            open(
                "/home/ec2-user/SageMaker/efs/data/CAMP/structures_train.pkl",
                "rb",
            )
        )
        len(train_structs)
        test_structs = pickle.load(
            open(
                "/home/ec2-user/SageMaker/efs/data/CAMP/structures_test.pkl",
                "rb",
            )
        )
        len(test_structs)
        column_names = [
            "PDB ID",
            "peptide chain ID",
            "peptide length",
            "number of atoms in peptide",
            "protein chain ID",
            "number of atoms in protein",
            "number of atom contacts between peptide and protein",
            "?",
            "peptide with nonstandard amino acid?",
            "resolution",
            "molecular type",
        ]
        # load metadata
        DATA_DIR = "/home/ec2-user/SageMaker/efs/data/PepBDB"
        metadata = os.path.join(DATA_DIR, "peptidelist.txt")
        df = pd.read_csv(
            metadata, header=None, delim_whitespace=True, names=column_names
        )
        data_list_train = prepare_pepbdb_data_list(train_structs, df)
        data_list_test = prepare_pepbdb_data_list(test_structs, df)

        # split train/val
        n_train = int(0.8 * len(data_list_train))

        if input_type == "complex":
            # Protein complex as input
            complex_featurizer = NaturalComplexFeaturizer()
            train_dataset = PepBDBComplexDataset(
                data_list_train[:n_train],
                featurizer=complex_featurizer,
                preprocess=True,
            )
            valid_dataset = PepBDBComplexDataset(
                data_list_train[n_train:],
                featurizer=complex_featurizer,
                preprocess=True,
            )
            test_dataset = PepBDBComplexDataset(
                data_list_test, featurizer=complex_featurizer, preprocess=True
            )
        elif input_type == "polypeptides":
            raise NotImplementedError
    return train_dataset, valid_dataset, test_dataset


def evaluate(model, data_loader):
    """Evaluate model on dataset and return metrics."""
    # make predictions on test set
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    MCC = torchmetrics.MatthewsCorrCoef(num_classes=2)
    AUPR = torchmetrics.AveragePrecision()
    AUROC = torchmetrics.AUROC()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.ndata["target"]
            train_mask = batch.ndata["mask"]
            probs = torch.sigmoid(logits[train_mask]).to("cpu")
            targets = targets[train_mask].to(torch.int).to("cpu")

            mcc = MCC(probs, targets)
            aupr = AUPR(probs, targets)
            auroc = AUROC(probs, targets)

    results = {
        "MCC": MCC.compute().item(),
        "AUPR": AUPR.compute().item(),
        "AUROC": AUROC.compute().item(),
    }
    return results


def main(args):
    # 1. Load data
    train_dataset, valid_dataset, test_dataset = get_datasets(
        name=args.dataset_name, input_type=args.input_type
    )
    print(
        "Data loaded:",
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )
    # 2. Prepare data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )
    # 3. Prepare model
    datum = train_dataset[0][0]
    dict_args = vars(args)
    model = init_model(datum=datum, num_outputs=1, **dict_args)
    # 4. Training model
    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping_patience
    )
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    # init pl.Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    # train
    trainer.fit(model, train_loader, valid_loader)
    print("Training finished")
    print(
        "checkpoint_callback.best_model_path:",
        checkpoint_callback.best_model_path,
    )
    # 5. Evaluation
    # load the best model
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
    )
    print("Testing performance on test set")
    scores = evaluate(model, test_loader)
    pprint(scores)
    # save scores to file
    json.dump(
        scores,
        open(os.path.join(trainer.log_dir, "scores.json"), "w"),
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="gvp",
        help="Choose from %s" % ", ".join(list(MODEL_CONSTRUCTORS.keys())),
    )
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    # add model specific args
    model_name = temp_args.model_name
    parser = MODEL_CONSTRUCTORS[model_name].add_model_specific_args(parser)

    # Additional params
    # dataset params
    parser.add_argument(
        "--dataset_name",
        help="dataset name",
        type=str,
        default="PepBDB",
    )
    parser.add_argument(
        "--input_type",
        help="data input type",
        type=str,
        default="complex",
    )
    # training hparams
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers used in DataLoader",
    )

    args = parser.parse_args()

    print("args:", args)
    # train
    main(args)
