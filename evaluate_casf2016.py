"""
Evaluate a trained pytorch-lightning model on the three tasks on CASF2016:
- Scoring => Spearman rho, R2
- Docking => top1, 2, 3 success rates 
- Screening => Average EF, success rates
"""

from train import get_datasets, evaluate_graph_regression, MODEL_CONSTRUCTORS
from evaluate import load_model_from_checkpoint

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from typing import Dict, List
import argparse
import os
import json
import glob
from pprint import pprint
from tqdm import tqdm
import torch
import numpy as np


def choose_best_pose(id_to_pred: Dict[str, float]) -> Dict[str, float]:
    pairs = ["_".join(k.split("_")[:-1]) for k in id_to_pred.keys()]
    pairs = sorted(list(set(pairs)))
    retval = {p: [] for p in pairs}
    for key in id_to_pred.keys():
        pair = "_".join(key.split("_")[:-1])
        retval[pair].append(id_to_pred[key])
    for key in retval.keys():
        retval[key] = min(retval[key])
    return retval


def predict(
    model,
    data_loader,
    model_name="gvp",
    use_energy_decoder=False,
    is_hetero=False,
):
    """Make predictions on data from the data_loader"""
    # make predictions on test set
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Move relevant tensors to GPU
            for key, val in batch.items():
                if key not in ("sample", "atom_to_residue", "smiles_strings"):
                    batch[key] = val.to(device)
            if model_name == "gvp":
                batch["graph"] = batch["graph"].to(device)
                if use_energy_decoder:
                    batch["sample"] = {
                        key: val.to(device)
                        for key, val in batch["sample"].items()
                    }
                    energies, _, _ = model(batch)
                    preds = energies.sum(-1).unsqueeze(-1)
                else:
                    _, preds = model(batch)
            elif model_name == "multistage-gvp":
                if use_energy_decoder:
                    batch["sample"] = {
                        key: val.to(device)
                        for key, val in batch["sample"].items()
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
                        key: val.to(device)
                        for key, val in batch["sample"].items()
                    }
                    if is_hetero:
                        energies, _, _ = model(
                            batch["protein_graph"],
                            batch["ligand_graph"],
                            batch["complex_graph"],
                            batch["sample"],
                            cal_der_loss=False,
                            atom_to_residue=batch["atom_to_residue"],
                            smiles_strings=batch["smiles_strings"],
                        )
                    else:
                        energies, _, _ = model(
                            batch["protein_graph"],
                            batch["ligand_graph"],
                            batch["complex_graph"],
                            batch["sample"],
                            cal_der_loss=False,
                            smiles_strings=batch["smiles_strings"],
                        )
                    preds = energies.sum(-1).unsqueeze(-1)
                else:
                    _, preds = model(
                        batch["protein_graph"],
                        batch["ligand_graph"],
                        batch["complex_graph"],
                        smiles_strings=batch["smiles_strings"],
                    )
            else:
                raise NotImplementedError
            preds = preds.to("cpu")
            preds = list(preds.numpy().reshape(-1))
            all_preds.extend(preds)
    return all_preds


def load_rmsd(rmsd_dir):
    """Load decoys docking RMSD from files"""
    rmsd_dir = os.path.join(rmsd_dir, "*_rmsd.dat")
    rmsd_filenames = glob.glob(rmsd_dir)
    id_to_rmsd = dict()
    for file in rmsd_filenames:
        with open(file, "r") as f:
            lines = f.readlines()[1:]
            lines = [line.split() for line in lines]
            lines = [[line[0], float(line[1])] for line in lines]
            dic = dict(lines)
            id_to_rmsd.update(dic)

    return id_to_rmsd


def load_screening_target_file(target_file):
    # Load target file
    target_file = "/home/ec2-user/SageMaker/efs/data/PIGNet/casf2016_benchmark/TargetInfo.dat"

    true_binder_list = []
    with open(target_file, "r") as f:
        lines = f.readlines()[9:]
        for line in lines:
            line = line.split()
            true_binder_list += [(line[0], elem) for elem in line[1:6]]
    return true_binder_list


def evaluate_docking(id_to_pred, id_to_rmsd):
    # modified from PIGNet/casf2016_benchmark/docking_power.py
    # calculate topn success
    pdbs = sorted(
        list(set(key.split()[0].split("_")[0] for key in id_to_pred))
    )
    topn_successed_pdbs = []
    for pdb in pdbs:
        selected_keys = [key for key in id_to_pred if pdb in key]
        pred = [id_to_pred[key] for key in selected_keys]
        pred, sorted_keys = zip(*sorted(zip(pred, selected_keys)))
        rmsd = [id_to_rmsd[key] for key in sorted_keys]
        topn_successed = []
        for topn in [1, 2, 3]:
            if min(rmsd[:topn]) < 2.0:
                topn_successed.append(1)
            else:
                topn_successed.append(0)
        topn_successed_pdbs.append(topn_successed)

    scores = {}
    for topn in [1, 2, 3]:
        successed = [success[topn - 1] for success in topn_successed_pdbs]
        success_rate = np.mean(successed)
        scores["success_rate_top%d" % topn] = success_rate
        print(round(success_rate, 3), end="\t")

    return scores


def evaluate_screening(id_to_pred, true_binder_list):
    ntb_top = []
    ntb_total = []
    high_affinity_success = []
    pdbs = sorted(list(set([key.split("_")[0] for key in id_to_pred.keys()])))
    for pdb in pdbs:
        selected_keys = [
            key for key in id_to_pred.keys() if key.split("_")[0] == pdb
        ]
        preds = [id_to_pred[key] for key in selected_keys]
        preds, selected_keys = zip(*sorted(zip(preds, selected_keys)))
        true_binders = [
            key
            for key in selected_keys
            if (key.split("_")[0], key.split("_")[1]) in true_binder_list
        ]
        ntb_top_pdb, ntb_total_pdb, high_affinity_success_pdb = [], [], []
        for topn in [0.01, 0.05, 0.1]:
            n = int(topn * len(selected_keys))
            top_keys = selected_keys[:n]
            n_top_true_binder = len(list(set(top_keys) & set(true_binders)))
            ntb_top_pdb.append(n_top_true_binder)
            ntb_total_pdb.append(len(true_binders) * topn)
            if f"{pdb}_{pdb}" in top_keys:
                high_affinity_success_pdb.append(1)
            else:
                high_affinity_success_pdb.append(0)
        ntb_top.append(ntb_top_pdb)
        ntb_total.append(ntb_total_pdb)
        high_affinity_success.append(high_affinity_success_pdb)

    scores = {}
    for i in range(3):
        ef = []
        for j in range(len(ntb_total)):
            if ntb_total[j][i] == 0:
                continue
            ef.append(ntb_top[j][i] / ntb_total[j][i])

        avg_ef = np.mean(ef)
        scores["avgEF_top_%d_pct" % (i + 1)] = avg_ef
        print(round(avg_ef, 3), end="\t")

    for i in range(3):
        success = []
        for j in range(len(ntb_total)):
            if high_affinity_success[j][i] > 0:
                success.append(1)
            else:
                success.append(0)

        success_rate = np.mean(success)
        scores["success_rate_top%d" % (i + 1)] = success_rate
        print(round(success_rate, 3), end="\t")
    return scores


def main(args):
    pl.seed_everything(42, workers=True)
    # 0. Prepare model
    model = load_model_from_checkpoint(args.checkpoint_path, args.model_name)
    if args.checkpoint_path.endswith(".ckpt"):
        checkpoint_path = os.path.dirname(
            os.path.dirname(args.checkpoint_path)
        )
    else:
        checkpoint_path = args.checkpoint_path
    # 1. Scoring data
    print("Performing scoring task...")
    scoring_dataset = get_datasets(
        name="PDBBind",
        input_type=args.input_type,
        data_dir=os.path.join(args.data_dir, "scoring"),
        test_only=True,
        residue_featurizer_name=args.residue_featurizer_name,
        use_energy_decoder=args.use_energy_decoder,
        intra_mol_energy=args.intra_mol_energy,
    )
    print(
        "Data loaded:",
        len(scoring_dataset),
    )
    scoring_data_loader = DataLoader(
        scoring_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=scoring_dataset.collate_fn,
    )
    scores = evaluate_graph_regression(
        model,
        scoring_data_loader,
        model_name=args.model_name,
        use_energy_decoder=args.use_energy_decoder,
        is_hetero=args.is_hetero,
    )
    pprint(scores)
    # save scores to file
    json.dump(
        scores,
        open(
            os.path.join(checkpoint_path, "casf2016_scoring_scores.json"),
            "w",
        ),
    )

    # 2. Docking data
    print("Performing docking task...")
    id_to_rmsd = load_rmsd(
        os.path.join(
            args.data_dir, "../../casf2016_benchmark/decoys_docking_rmsd"
        )
    )

    docking_dataset = get_datasets(
        name="PDBBind",
        input_type=args.input_type,
        data_dir=os.path.join(args.data_dir, "docking"),
        test_only=True,
        residue_featurizer_name=args.residue_featurizer_name,
        use_energy_decoder=args.use_energy_decoder,
        intra_mol_energy=args.intra_mol_energy,
    )
    print(
        "Data loaded:",
        len(docking_dataset),
    )
    docking_data_loader = DataLoader(
        docking_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=docking_dataset.collate_fn,
    )
    all_preds = predict(
        model,
        docking_data_loader,
        model_name=args.model_name,
        use_energy_decoder=args.use_energy_decoder,
        is_hetero=args.is_hetero,
    )
    id_to_pred = dict(zip(docking_dataset.keys, all_preds))

    docking_scores = evaluate_docking(id_to_pred, id_to_rmsd)
    # save scores to file
    json.dump(
        docking_scores,
        open(
            os.path.join(checkpoint_path, "casf2016_docking_scores.json"),
            "w",
        ),
    )
    # 3. Screening data
    print("Performing screening task...")
    true_binder_list = load_screening_target_file(
        os.path.join(args.data_dir, "../../casf2016_benchmark/TargetInfo.dat")
    )

    screening_dataset = get_datasets(
        name="PDBBind",
        input_type=args.input_type,
        data_dir=os.path.join(args.data_dir, "screening"),
        test_only=True,
        residue_featurizer_name=args.residue_featurizer_name,
        use_energy_decoder=args.use_energy_decoder,
        intra_mol_energy=args.intra_mol_energy,
    )
    print(
        "Data loaded:",
        len(screening_dataset),
    )
    screening_data_loader = DataLoader(
        screening_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=screening_dataset.collate_fn,
    )
    all_preds = predict(
        model,
        screening_data_loader,
        model_name=args.model_name,
        use_energy_decoder=args.use_energy_decoder,
        is_hetero=args.is_hetero,
    )
    id_to_pred = dict(zip(screening_dataset.keys, all_preds))
    screening_scores = evaluate_screening(id_to_pred, true_binder_list)
    # save scores to file
    json.dump(
        screening_scores,
        open(
            os.path.join(checkpoint_path, "casf2016_screening_scores.json"),
            "w",
        ),
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gvp",
        help="Choose from %s" % ", ".join(list(MODEL_CONSTRUCTORS.keys())),
    )
    parser.add_argument(
        "--input_type",
        help="data input type",
        type=str,
        default="complex",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="ptl checkpoint path like `lightning_logs/version_x`",
        required=True,
    )

    # dataset params
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        type=str,
        default="",
    )
    parser.add_argument(
        "--bs", type=int, default=64, help="batch size for test data"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers used in DataLoader",
    )
    # featurizer params
    parser.add_argument(
        "--residue_featurizer_name",
        help="name of the residue featurizer",
        type=str,
        default="MACCS",
    )

    parser.add_argument("--use_energy_decoder", action="store_true")
    parser.add_argument("--is_hetero", action="store_true")
    parser.add_argument("--intra_mol_energy", action="store_true")
    parser.set_defaults(
        use_energy_decoder=False, is_hetero=False, intra_mol_energy=False
    )

    args = parser.parse_args()

    print("args:", args)
    # evaluate
    main(args)
