from rdkit import Chem
from transformers import T5Tokenizer, T5EncoderModel
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
from torch.utils.data import DataLoader, random_split
import dgl

import torchmetrics

# custom imports
# from ppi.modules import GATModel, GVPModel
from ppi.model import (
    LitGVPModel,
    LitHGVPModel,
    LitMultiStageGVPModel,
    LitMultiStageHGVPModel,
)
from ppi.data import (
    PDBComplexDataset,
    PDBBigraphComplexDataset,
    PIGNetComplexDataset,
    PIGNetAtomicBigraphComplexDataset,
    PIGNetHeteroBigraphComplexDataset,
    PIGNetHeteroBigraphComplexDatasetForEnergyModel,
    PIGNetAtomicBigraphComplexEnergyDataset,
)
from ppi.data_utils import (
    get_residue_featurizer,
    NoncanonicalComplexFeaturizer,
    NoncanonicalBigraphComplexFeaturizer,
    PDBBindComplexFeaturizer,
    PIGNetHeteroBigraphComplexFeaturizer,
    PIGNetHeteroBigraphComplexFeaturizerForEnergyModel,
    PIGNetAtomicBigraphGeometricComplexFeaturizer,
    PIGNetAtomicBigraphPhysicalComplexFeaturizer,
)
from ppi.transfer import load_state_dict_to_model

# mapping model names to constructors
MODEL_CONSTRUCTORS = {
    "gvp": LitGVPModel,
    "hgvp": LitHGVPModel,
    "multistage-gvp": LitMultiStageGVPModel,
    "multistage-hgvp": LitMultiStageHGVPModel,
}


def init_model(
    datum=None,
    model_name="gvp",
    num_outputs=1,
    classify=False,
    pos_weight=None,
    **kwargs,
):
    if model_name in ["gvp", "hgvp"]:
        kwargs["node_h_dim"] = tuple(kwargs["node_h_dim"])
        kwargs["edge_h_dim"] = tuple(kwargs["edge_h_dim"])
        print("node_h_dim:", kwargs["node_h_dim"])
        print("edge_h_dim:", kwargs["edge_h_dim"])
        model = MODEL_CONSTRUCTORS[model_name](
            g=datum,
            num_outputs=num_outputs,
            classify=classify,
            pos_weight=pos_weight,
            **kwargs,
        )
    elif model_name == "multistage-gvp":
        protein_graph = datum["protein_graph"]
        ligand_graph = datum["ligand_graph"]
        complex_graph = datum["complex_graph"]

        kwargs["stage1_node_h_dim"] = tuple(kwargs["stage1_node_h_dim"])
        kwargs["stage1_edge_h_dim"] = tuple(kwargs["stage1_edge_h_dim"])
        print("stage1_node_h_dim:", kwargs["stage1_node_h_dim"])
        print("stage1_edge_h_dim:", kwargs["stage1_edge_h_dim"])

        kwargs["stage2_node_h_dim"] = tuple(kwargs["stage2_node_h_dim"])
        kwargs["stage2_edge_h_dim"] = tuple(kwargs["stage2_edge_h_dim"])
        print("stage2_node_h_dim:", kwargs["stage2_node_h_dim"])
        print("stage2_edge_h_dim:", kwargs["stage2_edge_h_dim"])

        # Protein inputs
        protein_node_in_dim = (
            protein_graph.ndata["node_s"].shape[1],
            protein_graph.ndata["node_v"].shape[1],
        )
        protein_edge_in_dim = (
            protein_graph.edata["edge_s"].shape[1],
            protein_graph.edata["edge_v"].shape[1],
        )

        # Ligand inputs
        ligand_node_in_dim = (
            ligand_graph.ndata["node_s"].shape[1],
            ligand_graph.ndata["node_v"].shape[1],
        )
        ligand_edge_in_dim = (
            ligand_graph.edata["edge_s"].shape[1],
            ligand_graph.edata["edge_v"].shape[1],
        )

        # Complex inputs
        complex_edge_in_dim = (
            complex_graph.edata["edge_s"].shape[1],
            complex_graph.edata["edge_v"].shape[1],
        )

        model = MODEL_CONSTRUCTORS[model_name](
            protein_node_in_dim=protein_node_in_dim,
            protein_edge_in_dim=protein_edge_in_dim,
            ligand_node_in_dim=ligand_node_in_dim,
            ligand_edge_in_dim=ligand_edge_in_dim,
            complex_edge_in_dim=complex_edge_in_dim,
            num_outputs=num_outputs,
            classify=classify,
            **kwargs,
        )
    elif model_name == "multistage-hgvp":
        protein_graph = datum["protein_graph"]
        ligand_graph = datum["ligand_graph"]
        complex_graph = datum["complex_graph"]

        kwargs["stage1_node_h_dim"] = tuple(kwargs["stage1_node_h_dim"])
        kwargs["stage1_edge_h_dim"] = tuple(kwargs["stage1_edge_h_dim"])
        print("stage1_node_h_dim:", kwargs["stage1_node_h_dim"])
        print("stage1_edge_h_dim:", kwargs["stage1_edge_h_dim"])

        kwargs["stage2_node_h_dim"] = tuple(kwargs["stage2_node_h_dim"])
        kwargs["stage2_edge_h_dim"] = tuple(kwargs["stage2_edge_h_dim"])
        print("stage2_node_h_dim:", kwargs["stage2_node_h_dim"])
        print("stage2_edge_h_dim:", kwargs["stage2_edge_h_dim"])

        # Protein inputs
        protein_node_in_dim = (
            protein_graph.ndata["node_s"].shape[1],
            protein_graph.ndata["node_v"].shape[1],
        )
        protein_edge_in_dim = (
            protein_graph.edata["edge_s"].shape[1],
            protein_graph.edata["edge_v"].shape[1],
        )

        # Ligand inputs
        ligand_node_in_dim = (
            ligand_graph.ndata["node_s"].shape[1],
            ligand_graph.ndata["node_v"].shape[1],
        )
        ligand_edge_in_dim = (
            ligand_graph.edata["edge_s"].shape[1],
            ligand_graph.edata["edge_v"].shape[1],
        )

        # Complex inputs
        complex_edge_in_dim = (
            complex_graph.edata["edge_s"].shape[1],
            complex_graph.edata["edge_v"].shape[1],
        )

        # if kwargs["is_hetero"]:
        #     model = MODEL_CONSTRUCTORS[model_name](
        #         g_protein=protein_graph,
        #         ligand_node_in_dim=ligand_node_in_dim,
        #         ligand_edge_in_dim=ligand_edge_in_dim,
        #         complex_edge_in_dim=complex_edge_in_dim,
        #         num_outputs=num_outputs,
        #         classify=classify,
        #         **kwargs,
        #     )
        # else:
        model = MODEL_CONSTRUCTORS[model_name](
                g_protein=protein_graph,
                g_ligand=ligand_graph,
                complex_edge_in_dim=complex_edge_in_dim,
                num_outputs=num_outputs,
                classify=classify,
                **kwargs,
            )
    else:
        model = MODEL_CONSTRUCTORS[model_name](
            in_feats=datum.ndata["node_s"].shape[1],
            num_outputs=num_outputs,
            classify=classify,
            pos_weight=pos_weight,
            **kwargs,
        )

    return model


def get_datasets(
    name="PDBBind",
    input_type="complex",
    data_dir="",
    test_only=False,
    residue_featurizer_name="MACCS",
    use_energy_decoder=False,
    intra_mol_energy=False,
    data_suffix="full",
    binary_cutoff=None,
    add_noise=0.0,
):
    # initialize residue featurizer
    if "grad" in residue_featurizer_name:
        # Do not init residue_featurizer if it involes grad
        # This will allow joint training of residue_featurizer with the
        # model
        residue_featurizer = None
    else:
        residue_featurizer = get_residue_featurizer(residue_featurizer_name)
    # initialize complex featurizer based on dataset type
    if name in ("Propedia", "ProtCID"):
        # load Propedia metadata
        if input_type == "complex":
            featurizer = NoncanonicalComplexFeaturizer(residue_featurizer)
            test_df = pd.read_csv(
                os.path.join(data_dir, f"test_{data_suffix}.csv")
            )
            test_dataset = PDBComplexDataset(
                test_df,
                data_dir,
                featurizer=featurizer,
            )
            if not test_only:
                train_df = pd.read_csv(
                    os.path.join(data_dir, f"train_{data_suffix}.csv")
                ).sample(frac=1)
                n_train = int(0.8 * train_df.shape[0])
                featurizer = NoncanonicalComplexFeaturizer(
                    residue_featurizer, add_noise=add_noise
                )
                train_dataset = PDBComplexDataset(
                    train_df.iloc[:n_train],
                    data_dir,
                    featurizer=featurizer,
                )
                valid_dataset = PDBComplexDataset(
                    train_df.iloc[n_train:],
                    data_dir,
                    featurizer=featurizer,
                )
        elif input_type == "multistage-complex":
            featurizer = NoncanonicalBigraphComplexFeaturizer(residue_featurizer)
            test_df = pd.read_csv(
                os.path.join(data_dir, f"test_{data_suffix}.csv")
            )
            test_dataset = PDBBigraphComplexDataset(
                test_df,
                data_dir,
                featurizer=featurizer,
            )
            if not test_only:
                train_df = pd.read_csv(
                    os.path.join(data_dir, f"train_{data_suffix}.csv")
                ).sample(frac=1)
                n_train = int(0.8 * train_df.shape[0])
                featurizer = NoncanonicalBigraphComplexFeaturizer(
                    residue_featurizer, add_noise=add_noise
                )
                train_dataset = PDBBigraphComplexDataset(
                    train_df.iloc[:n_train],
                    data_dir,
                    featurizer=featurizer,
                )
                valid_dataset = PDBBigraphComplexDataset(
                    train_df.iloc[n_train:],
                    data_dir,
                    featurizer=featurizer,
                )
        elif input_type == "polypeptides":
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif name == "PDBBind":
        # PIGNet parsed PDBBind datasets
        # read labels
        with open(os.path.join(data_dir, "pdb_to_affinity.txt")) as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]
            id_to_y = {l[0]: float(l[1]) for l in lines}

        with open(os.path.join(data_dir, "keys/test_keys.pkl"), "rb") as f:
            test_keys = pickle.load(f)

        # featurizer for PDBBind
        if input_type == "complex":
            featurizer = PDBBindComplexFeaturizer(
                residue_featurizer, count_atoms=use_energy_decoder
            )
            test_dataset = PIGNetComplexDataset(
                test_keys,
                data_dir,
                id_to_y,
                featurizer,
                compute_energy=use_energy_decoder,
                intra_mol_energy=intra_mol_energy,
            )
            if not test_only:
                with open(
                    os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                ) as f:
                    train_keys = pickle.load(f)
                n_train = int(0.8 * len(train_keys))
                train_dataset = PIGNetComplexDataset(
                    train_keys[:n_train],
                    data_dir,
                    id_to_y,
                    featurizer,
                    compute_energy=use_energy_decoder,
                    intra_mol_energy=intra_mol_energy,
                )
                valid_dataset = PIGNetComplexDataset(
                    train_keys[n_train:],
                    data_dir,
                    id_to_y,
                    featurizer,
                    compute_energy=use_energy_decoder,
                    intra_mol_energy=intra_mol_energy,
                )
        elif input_type == "multistage-hetero":
            if use_energy_decoder:
                featurizer = (
                    PIGNetHeteroBigraphComplexFeaturizerForEnergyModel(
                        residue_featurizer=residue_featurizer
                    )
                )
                test_dataset = PIGNetHeteroBigraphComplexDatasetForEnergyModel(
                    test_keys, data_dir, id_to_y, featurizer
                )
                if not test_only:
                    with open(
                        os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                    ) as f:
                        train_keys = pickle.load(f)
                    n_train = int(0.8 * len(train_keys))
                    train_dataset = (
                        PIGNetHeteroBigraphComplexDatasetForEnergyModel(
                            train_keys[:n_train], data_dir, id_to_y, featurizer
                        )
                    )
                    valid_dataset = (
                        PIGNetHeteroBigraphComplexDatasetForEnergyModel(
                            train_keys[n_train:], data_dir, id_to_y, featurizer
                        )
                    )
            else:
                featurizer = PIGNetHeteroBigraphComplexFeaturizer(
                    residue_featurizer=residue_featurizer
                )
                test_dataset = PIGNetHeteroBigraphComplexDataset(
                    test_keys, data_dir, id_to_y, featurizer
                )
                if not test_only:
                    with open(
                        os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                    ) as f:
                        train_keys = pickle.load(f)
                    n_train = int(0.8 * len(train_keys))
                    train_dataset = PIGNetHeteroBigraphComplexDataset(
                        train_keys[:n_train], data_dir, id_to_y, featurizer
                    )
                    valid_dataset = PIGNetHeteroBigraphComplexDataset(
                        train_keys[n_train:], data_dir, id_to_y, featurizer
                    )
        elif input_type == "multistage-geometric":
            if use_energy_decoder:
                featurizer = PIGNetAtomicBigraphGeometricComplexFeaturizer(
                    residue_featurizer=None, return_physics=True
                )
                test_dataset = PIGNetAtomicBigraphComplexEnergyDataset(
                    test_keys, data_dir, id_to_y, featurizer
                )
                if not test_only:
                    with open(
                        os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                    ) as f:
                        train_keys = pickle.load(f)
                    n_train = int(0.8 * len(train_keys))
                    train_dataset = PIGNetAtomicBigraphComplexEnergyDataset(
                        train_keys[:n_train], data_dir, id_to_y, featurizer
                    )
                    valid_dataset = PIGNetAtomicBigraphComplexEnergyDataset(
                        train_keys[n_train:], data_dir, id_to_y, featurizer
                    )
            else:
                featurizer = PIGNetAtomicBigraphGeometricComplexFeaturizer(
                    residue_featurizer=None
                )
                test_dataset = PIGNetAtomicBigraphComplexDataset(
                    test_keys, data_dir, id_to_y, featurizer
                )
                if not test_only:
                    with open(
                        os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                    ) as f:
                        train_keys = pickle.load(f)
                    n_train = int(0.8 * len(train_keys))
                    train_dataset = PIGNetAtomicBigraphComplexDataset(
                        train_keys[:n_train], data_dir, id_to_y, featurizer
                    )
                    valid_dataset = PIGNetAtomicBigraphComplexDataset(
                        train_keys[n_train:], data_dir, id_to_y, featurizer
                    )
        elif input_type == "multistage-physical":
            if use_energy_decoder:
                featurizer = PIGNetAtomicBigraphPhysicalComplexFeaturizer(
                    residue_featurizer=None, return_physics=True
                )
                test_dataset = PIGNetAtomicBigraphComplexEnergyDataset(
                    test_keys, data_dir, id_to_y, featurizer
                )
                if not test_only:
                    with open(
                        os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                    ) as f:
                        train_keys = pickle.load(f)
                    n_train = int(0.8 * len(train_keys))
                    train_dataset = PIGNetAtomicBigraphComplexEnergyDataset(
                        train_keys[:n_train], data_dir, id_to_y, featurizer
                    )
                    valid_dataset = PIGNetAtomicBigraphComplexEnergyDataset(
                        train_keys[n_train:], data_dir, id_to_y, featurizer
                    )
            else:
                featurizer = PIGNetAtomicBigraphPhysicalComplexFeaturizer(
                    residue_featurizer=None
                )
                test_dataset = PIGNetAtomicBigraphComplexDataset(
                    test_keys, data_dir, id_to_y, featurizer
                )
                if not test_only:
                    with open(
                        os.path.join(data_dir, "keys/train_keys.pkl"), "rb"
                    ) as f:
                        train_keys = pickle.load(f)
                    n_train = int(0.8 * len(train_keys))
                    train_dataset = PIGNetAtomicBigraphComplexDataset(
                        train_keys[:n_train], data_dir, id_to_y, featurizer
                    )
                    valid_dataset = PIGNetAtomicBigraphComplexDataset(
                        train_keys[n_train:], data_dir, id_to_y, featurizer
                    )

        else:
            raise NotImplementedError
    if not test_only:
        return train_dataset, valid_dataset, test_dataset
    else:
        return test_dataset

def evaluate_node_classification(model, data_loader):
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
            logits, _ = model(batch)
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


def predict_step(
    model,
    batch,
    device,
    model_name="gvp",
    use_energy_decoder=False,
    is_hetero=False,
):
    """Make prediction on one batch of data"""
    # Move relevant tensors to GPU
    for key, val in batch.items():
        if key not in ("sample", "atom_to_residue", "smiles_strings", "protein_smiles_strings", "ligand_smiles_strings"):
            batch[key] = val.to(device)
    if model_name in ("gvp", "hgvp"):
        batch["graph"] = batch["graph"].to(device)
        if use_energy_decoder:
            batch["sample"] = {
                key: val.to(device) for key, val in batch["sample"].items()
            }
            energies, _, _ = model(batch)
            preds = energies.sum(-1).unsqueeze(-1)
        else:
            _, preds = model(batch)
    elif model_name == "multistage-gvp":
        if use_energy_decoder:
            batch["sample"] = {
                key: val.to(device) for key, val in batch["sample"].items()
            }
            if is_hetero:
                energies, _, _ = model(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    cal_der_loss=False,
                    atom_to_residue=batch["atom_to_residue"],
                )
            else:
                energies, _, _ = model(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    cal_der_loss=False,
                )
            preds = energies.sum(-1).unsqueeze(-1)
        else:
            _, preds = model(
                batch["protein_graph"],
                batch["ligand_graph"],
                batch["complex_graph"],
            )
    elif model_name == "multistage-hgvp":
        if use_energy_decoder:
            batch["sample"] = {
                key: val.to(device) for key, val in batch["sample"].items()
            }
            if is_hetero:
                energies, _, _ = model(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    cal_der_loss=False,
                    atom_to_residue=batch["atom_to_residue"],
                    protein_smiles_strings=batch["protein_smiles_strings"],
                    ligand_smiles_strings=batch["ligand_smiles_strings"],
                )
            else:
                energies, _, _ = model(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    cal_der_loss=False,
                    protein_smiles_strings=batch["protein_smiles_strings"],
                    ligand_smiles_strings=batch["ligand_smiles_strings"],
                )
            preds = energies.sum(-1).unsqueeze(-1)
        else:
            _, preds = model(
                batch["protein_graph"],
                batch["ligand_graph"],
                batch["complex_graph"],
                protein_smiles_strings=batch["protein_smiles_strings"],
                ligand_smiles_strings=batch["ligand_smiles_strings"],
            )
    else:
        raise NotImplementedError
    return preds


def evaluate_graph_regression(
    model,
    data_loader,
    model_name="gvp",
    use_energy_decoder=False,
    is_hetero=False,
):
    """Evaluate model on dataset and return metrics for graph-level regression."""
    # make predictions on test set
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    R2Score = torchmetrics.R2Score()
    SpearmanCorrCoef = torchmetrics.SpearmanCorrCoef()
    MSE = torchmetrics.MeanSquaredError()
    with torch.no_grad():
        for batch in data_loader:
            preds = predict_step(
                model,
                batch,
                device,
                model_name=model_name,
                use_energy_decoder=use_energy_decoder,
                is_hetero=is_hetero,
            )
            preds = preds.to("cpu")
            targets = batch["g_targets"].to("cpu")

            r2 = R2Score(preds, targets)
            rho = SpearmanCorrCoef(preds, targets)
            mse = MSE(preds, targets)

    results = {
        "R2": R2Score.compute().item(),
        "rho": SpearmanCorrCoef.compute().item(),
        "MSE": MSE.compute().item(),
    }
    return results


def evaluate_graph_classification(
    model,
    data_loader,
    model_name="gvp",
    use_energy_decoder=False,
    is_hetero=False,
):
    """Evaluate model on datasets and return metrics for graph-level
    binary classification."""
    # make predictions on test set
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    AUROC = torchmetrics.AUROC()
    AP = torchmetrics.AveragePrecision()
    MCC = torchmetrics.MatthewsCorrCoef(num_classes=2)
    with torch.no_grad():
        for batch in data_loader:
            preds = predict_step(
                model,
                batch,
                device,
                model_name=model_name,
                use_energy_decoder=use_energy_decoder,
                is_hetero=is_hetero,
            )
            preds = preds.to("cpu")
            targets = batch["g_targets"].to("cpu").to(torch.int8)

            auroc = AUROC(preds, targets)
            ap = AP(preds, targets)
            mcc = MCC(preds.sigmoid(), targets)
    results = {
        "AUROC": AUROC.compute().item(),
        "AP": AP.compute().item(),
        "MCC": MCC.compute().item(),
    }
    return results


def main(args):
    pl.seed_everything(args.random_seed, workers=True)
    # 1. Load data
    train_dataset, valid_dataset, test_dataset = get_datasets(
        name=args.dataset_name,
        input_type=args.input_type,
        data_dir=args.data_dir,
        residue_featurizer_name=args.residue_featurizer_name,
        use_energy_decoder=args.use_energy_decoder,
        intra_mol_energy=args.intra_mol_energy,
        data_suffix=args.data_suffix,
        binary_cutoff=args.binary_cutoff,
        add_noise=args.add_noise,
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
        collate_fn=test_dataset.collate_fn,
        persistent_workers=args.persistent_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        persistent_workers=args.persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        persistent_workers=args.persistent_workers,
    )
    # 3. Prepare model
    if args.dataset_name in ["PDBBind", "ProtCID"]:
        if args.model_name in ["gvp", "hgvp"]:
            datum = train_dataset[0]["graph"]
        elif args.model_name in ["multistage-gvp", "multistage-hgvp"]:
            datum = train_dataset[0]
        else:
            raise NotImplementedError
    else:
        datum = train_dataset[0]["graph"]
    dict_args = vars(args)
    pos_weight = getattr(train_dataset, "pos_weight", None)
    classify = pos_weight is not None
    model = init_model(
        datum=datum,
        num_outputs=1,
        classify=classify,
        pos_weight=pos_weight,
        **dict_args,
    )

    if args.pretrained_weights:
        # load pretrained weights
        checkpoint = torch.load(
            args.pretrained_weights, map_location=torch.device("cpu")
        )
        load_state_dict_to_model(model, checkpoint["state_dict"])

    # 4. Training model
    # callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping_patience
    )
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)
    # init pl.Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    log_dir = trainer.log_dir
    # train
    trainer.fit(model, train_loader, valid_loader)
    print("Training finished")
    # save train args
    json.dump(dict_args, open(os.path.join(log_dir, "train_args.json"), "w"))
    # 5. Evaluation
    # load the best model
    if checkpoint_callback.best_model_path:
        print(
            "checkpoint_callback.best_model_path:",
            checkpoint_callback.best_model_path,
        )
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path,
        )
    print("Testing performance on test set")
    if args.dataset_name == "PDBBind":
        eval_func = evaluate_graph_regression
    elif classify:
        eval_func = evaluate_graph_classification

    scores = eval_func(
        model,
        test_loader,
        model_name=args.model_name,
        use_energy_decoder=args.use_energy_decoder,
        is_hetero=args.is_hetero,
    )
    pprint(scores)
    # save scores to file
    log_dir = os.path.join(
        os.path.dirname(checkpoint_callback.best_model_path), "../"
    )
    json.dump(
        scores,
        open(os.path.join(log_dir, "scores.json"), "w"),
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
    parser.add_argument(
        "--random_seed", help="global random seed", type=int, default=42
    )
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
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        type=str,
        default="",
    )
    parser.add_argument(
        "--data_suffix",
        help="used to distinguish different verions of the same dataset",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--binary_cutoff",
        help="used to convert PDBBind to a binary classification problem",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--add_noise",
        help="to add Gaussian noise to atom coordinates in train/valid sets",
        type=float,
        default=0.0,
    )
    # featurizer params
    parser.add_argument(
        "--residue_featurizer_name",
        help="name of the residue featurizer",
        type=str,
        default="MACCS",
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
    parser.add_argument(
        "--persistent_workers",
        type=bool,
        default=False,
        help="persistent_workers in DataLoader",
    )

    parser.add_argument("--use_energy_decoder", action="store_true")
    parser.add_argument("--is_hetero", action="store_true")
    parser.add_argument("--intra_mol_energy", action="store_true")
    parser.set_defaults(
        use_energy_decoder=False, is_hetero=False, intra_mol_energy=False
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="path to pretrained weights to initialize model",
    )

    args = parser.parse_args()

    print("args:", args)
    # train
    main(args)
