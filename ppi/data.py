# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Pytorch dataset classes from PPI prediction.
"""
from ppi.data_utils.pignet_featurizers import mol_to_feature
from rdkit import Chem
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch
import torch.utils.data as data
import dgl
from tqdm import tqdm

import numpy as np
import pickle
from Bio.PDB import PDBParser, MMCIFParser

# custom modules
from ppi.data_utils import (
    remove_nan_residues,
    mol_to_pdb_structure,
    residue_to_mol,
    parse_structure,
)


def check_dimension(tensors: List[Any]) -> Any:
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)

    return np.max(size, 0)


def collate_tensor(tensor: Any, max_tensor: Any, batch_idx: int) -> Any:
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor

    return max_tensor


def tensor_collate_fn(batch: List[Any]) -> Dict[str, Any]:
    batch_items = [it for e in batch for it in e.items()]
    dim_dict = dict()
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_element = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_element]
    for i, k in enumerate(total_key):
        value_list = [
            v for j, v in enumerate(total_value) if j % n_element == i
        ]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(
                np.array([batch_size, *check_dimension(value_list)])
            )
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))

    ret_dict = {}
    for j in range(batch_size):
        if batch[j] is None:
            continue
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value

    return ret_dict


class BasePPIDataset(data.Dataset):
    """Dataset for the Base Protein Graph."""

    def __init__(self, preprocess=False):
        self.processed_data = pd.Series([None] * len(self))
        if preprocess:
            print("Preprocessing data...")
            self._preprocess_all()

    def __getitem__(self, i):
        if self.processed_data[i] is None:
            # if not processed, process this instance and update
            self.processed_data[i] = self._preprocess(i)
        return self.processed_data[i]

    def _preprocess(self, complex):
        raise NotImplementedError

    def _preprocess_all(self):
        """Preprocess all the records in `data_list` with `_preprocess"""
        for i in tqdm(range(len(self.processed_data))):
            self.processed_data[i] = self._preprocess(i)


class PDBComplexDataset(BasePPIDataset):
    """
    To work with Propedia and ProtCID data, where each individual sample is a
    PDB complex file.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        path_to_data_files: str,
        featurizer: object,
        compute_energy=False,
        intra_mol_energy=False,
        **kwargs
    ):
        self.meta_df = meta_df
        self.path = path_to_data_files
        self.pdb_parser = PDBParser(
            QUIET=True,
            PERMISSIVE=True,
        )
        self.cif_parser = MMCIFParser(QUIET=True)
        self.featurizer = featurizer
        self.compute_energy = compute_energy
        self.intra_mol_energy = intra_mol_energy
        super(PDBComplexDataset, self).__init__(**kwargs)

    def __len__(self) -> int:
        return self.meta_df.shape[0]

    def _preprocess(self, idx: int) -> Dict[str, Any]:
        row = self.meta_df.iloc[idx]
        structure = parse_structure(
            self.pdb_parser,
            self.cif_parser,
            name=str(idx),
            file_path=os.path.join(self.path, row["pdb_file"]),
        )
        for chain in structure.get_chains():
            if chain.id == row["receptor_chain_id"]:
                protein = chain
            elif chain.id == row["ligand_chain_id"]:
                ligand = chain
        sample = self.featurizer.featurize(
            {"ligand": ligand, "protein": protein}
        )
        sample["target"] = row["label"]
        if self.compute_energy:
            ligand_mol = residue_to_mol(ligand, sanitize=False)
            protein_mol = residue_to_mol(protein, sanitize=False)
            physics = mol_to_feature(
                ligand_mol, protein_mol, compute_full=self.intra_mol_energy
            )
            sample["physics"] = physics
        return sample

    @property
    def pos_weight(self) -> torch.Tensor:
        """To compute the weight of the positive class, assuming binary
        classification"""
        class_sizes = self.meta_df["label"].value_counts()
        pos_weights = np.mean(class_sizes) / class_sizes
        pos_weights = torch.from_numpy(pos_weights.values.astype(np.float32))
        return pos_weights[1] / pos_weights[0]

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        graphs = []
        smiles_strings = []
        g_targets = []
        physics = []
        for rec in samples:
            graphs.append(rec["graph"])
            g_targets.append(rec["target"])
            if "smiles_strings" in rec:
                smiles_strings.extend(rec["smiles_strings"])
            if self.compute_energy:
                physics.append(rec["physics"])
        res = {
            "graph": dgl.batch(graphs),
            "g_targets": torch.tensor(g_targets)
            .to(torch.float32)
            .unsqueeze(-1),
            "smiles_strings": smiles_strings,
        }
        if self.compute_energy:
            res["sample"] = tensor_collate_fn(physics)
        return res


class PIGNetComplexDataset(data.Dataset):
    """
    To work with preprocessed pickles sourced from PDBBind dataset by the
    PIGNet paper.
    Modified from https://github.com/ACE-KAIST/PIGNet/blob/main/dataset.py
    """

    def __init__(
        self,
        keys: List[str],
        data_dir: str,
        id_to_y: Dict[str, float],
        featurizer: object,
        compute_energy=False,
        intra_mol_energy=False,
        binary_cutoff=None,
    ):
        self.keys = np.array(keys).astype(np.unicode_)
        self.data_dir = data_dir
        self.id_to_y = pd.Series(id_to_y)
        self.featurizer = featurizer
        self.processed_data = pd.Series([None] * len(self))
        self.compute_energy = compute_energy
        self.intra_mol_energy = intra_mol_energy
        self.binary_cutoff = binary_cutoff

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.processed_data[idx] is None:
            self.processed_data[idx] = self._preprocess(idx)
        return self.processed_data[idx]

    def _preprocess_all(self):
        """Preprocess all the records in `data_list` with `_preprocess"""
        for i in tqdm(range(len(self))):
            self.processed_data[i] = self._preprocess(i)

    def _preprocess(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        with open(os.path.join(self.data_dir, "data", key), "rb") as f:
            m1, _, m2, _ = pickle.load(f)

        if type(m2) is Chem.rdchem.Mol:
            protein_mol = m2
            protein_pdb = mol_to_pdb_structure(m2)
        else:
            protein_pdb = m2
            protein_mol = None

        sample = self.featurizer.featurize(
            {
                "ligand": m1,
                "protein": protein_pdb,
            }
        )
        if self.binary_cutoff is None:
            sample["affinity"] = self.id_to_y[key] * -1.36
        else:
            # convert to a binary classification problem:
            sample["affinity"] = self.id_to_y[key] >= self.binary_cutoff
        sample["key"] = key
        if self.compute_energy:
            if protein_mol is None:
                protein_mol = residue_to_mol(protein_pdb, sanitize=False)
            physics = mol_to_feature(
                m1, protein_mol, compute_full=self.intra_mol_energy
            )
            sample["physics"] = physics
        return sample

    @property
    def pos_weight(self) -> torch.Tensor:
        """To compute the weight of the positive class, assuming binary
        classification"""
        if self.binary_cutoff is None:
            return None
        else:
            affinities = self.id_to_y.loc[self.keys] > self.binary_cutoff
            class_sizes = affinities.astype(int).value_counts()
            pos_weights = np.mean(class_sizes) / class_sizes
            pos_weights = torch.from_numpy(
                pos_weights.values.astype(np.float32)
            )
            return pos_weights[1] / pos_weights[0]

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        graphs = []
        smiles_strings = []
        g_targets = []
        physics = []
        for rec in samples:
            graphs.append(rec["graph"])
            g_targets.append(rec["affinity"])
            if "smiles_strings" in rec:
                smiles_strings.extend(rec["smiles_strings"])
            if self.compute_energy:
                physics.append(rec["physics"])
        res = {
            "graph": dgl.batch(graphs),
            "g_targets": torch.tensor(g_targets)
            .to(torch.float32)
            .unsqueeze(-1),
            "smiles_strings": smiles_strings,
        }
        if self.compute_energy:
            res["sample"] = tensor_collate_fn(physics)
        return res


class PIGNetHeteroBigraphComplexDataset(data.Dataset):
    """
    To work with preprocessed pickles sourced from PDBBind dataset by the
    PIGNet paper.
    Modified from https://github.com/ACE-KAIST/PIGNet/blob/main/dataset.py
    """

    def __init__(
        self,
        keys: List[str],
        data_dir: str,
        id_to_y: Dict[str, float],
        featurizer: object,
    ):
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        with open(os.path.join(self.data_dir, "data", key), "rb") as f:
            m1, _, m2, _ = pickle.load(f)
        if type(m2) is Chem.rdchem.Mol:
            m2 = mol_to_pdb_structure(m2)

        if self.featurizer.residue_featurizer:
            (
                protein_graph,
                ligand_graph,
                complex_graph,
            ) = self.featurizer.featurize(
                {
                    "ligand": m1,
                    "protein": m2,
                }
            )
            sample = {
                "protein_graph": protein_graph,
                "ligand_graph": ligand_graph,
                "complex_graph": complex_graph,
            }
        else:
            (
                protein_graph,
                ligand_graph,
                complex_graph,
                protein_smiles_strings,
                ligand_smiles,
            ) = self.featurizer.featurize(
                {
                    "ligand": m1,
                    "protein": m2,
                }
            )
            sample = {
                "protein_graph": protein_graph,
                "ligand_graph": ligand_graph,
                "complex_graph": complex_graph,
                "protein_smiles_strings": protein_smiles_strings,
                "ligand_smiles_strings": None,
                "ligand_smiles": ligand_smiles,
            }
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        (
            protein_graphs,
            ligand_graphs,
            complex_graphs,
            protein_smiles_strings,
            ligand_smiles,
        ) = ([], [], [], [], [])
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            g_targets.append(rec["affinity"])
            if "protein_smiles_strings" in rec:
                protein_smiles_strings.extend(rec["protein_smiles_strings"])
            if "ligand_smiles" in rec:
                ligand_smiles.append(rec["ligand_smiles"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "g_targets": torch.tensor(g_targets).unsqueeze(-1),
            "protein_smiles_strings": protein_smiles_strings,
            "ligand_smiles_strings": None,
            "ligand_smiles": ligand_smiles,
        }


class PIGNetHeteroBigraphComplexDatasetForEnergyModel(data.Dataset):
    """
    To work with preprocessed pickles sourced from PDBBind dataset by the
    PIGNet paper.
    Modified from https://github.com/ACE-KAIST/PIGNet/blob/main/dataset.py
    """

    def __init__(
        self,
        keys: List[str],
        data_dir: str,
        id_to_y: Dict[str, float],
        featurizer: object,
    ):
        self.keys = keys
        self.data_dir = data_dir
        self.id_to_y = id_to_y
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        with open(os.path.join(self.data_dir, "data", key), "rb") as f:
            m1, _, m2, _ = pickle.load(f)

        if type(m2) is Chem.rdchem.Mol:
            protein_atoms = m2
            protein_residues = mol_to_pdb_structure(m2)
        else:
            protein_residues = m2
            protein_atoms = residue_to_mol(m2)

        if self.featurizer.residue_featurizer:
            (
                protein_graph,
                ligand_graph,
                complex_graph,
                physics,
                atom_to_residue,
            ) = self.featurizer.featurize(
                {
                    "ligand": m1,
                    "protein_atoms": protein_atoms,
                    "protein_residues": protein_residues,
                }
            )
            sample = {
                "protein_graph": protein_graph,
                "ligand_graph": ligand_graph,
                "complex_graph": complex_graph,
                "sample": physics,
                "atom_to_residue": atom_to_residue,
            }
        else:
            (
                protein_graph,
                ligand_graph,
                complex_graph,
                physics,
                atom_to_residue,
                smiles_strings,
                ligand_smiles,
            ) = self.featurizer.featurize(
                {
                    "ligand": m1,
                    "protein_atoms": protein_atoms,
                    "protein_residues": protein_residues,
                }
            )
            sample = {
                "protein_graph": protein_graph,
                "ligand_graph": ligand_graph,
                "complex_graph": complex_graph,
                "sample": physics,
                "atom_to_residue": atom_to_residue,
                "protein_smiles_strings": smiles_strings,
                "ligand_smiles_strings": None,
                "ligand_smiles": ligand_smiles,
            }
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        (
            protein_graphs,
            ligand_graphs,
            complex_graphs,
            physics,
            atom_to_residues,
            protein_smiles_strings,
            ligand_smiles,
        ) = ([], [], [], [], [], [], [])
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            physics.append(rec["sample"])
            atom_to_residues.append(rec["atom_to_residue"])
            g_targets.append(rec["affinity"])
            if "protein_smiles_strings" in rec:
                protein_smiles_strings.extend(rec["protein_smiles_strings"])
            if "ligand_smiles" in rec:
                ligand_smiles.append(rec["ligand_smiles"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "sample": tensor_collate_fn(physics),
            "atom_to_residue": atom_to_residues,
            "g_targets": torch.tensor(g_targets).unsqueeze(-1),
            "protein_smiles_strings": protein_smiles_strings,
            "ligand_smiles_strings": None,
            "ligand_smiles": ligand_smiles,
        }


class PDBBigraphComplexDataset(BasePPIDataset):
    """
    To work with Propedia and ProtCID data, where each individual sample is a
    PDB complex file.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        path_to_data_files: str,
        featurizer: object,
        **kwargs
    ):
        self.meta_df = meta_df
        self.path = path_to_data_files
        self.pdb_parser = PDBParser(
            QUIET=True,
            PERMISSIVE=True,
        )
        self.cif_parser = MMCIFParser(QUIET=True)
        self.featurizer = featurizer
        super(PDBBigraphComplexDataset, self).__init__(**kwargs)

    def __len__(self) -> int:
        return self.meta_df.shape[0]

    def _preprocess(self, idx: int) -> Dict[str, Any]:
        row = self.meta_df.iloc[idx]
        structure = parse_structure(
            self.pdb_parser,
            self.cif_parser,
            name=str(idx),
            file_path=os.path.join(self.path, row["pdb_file"]),
        )
        for chain in structure.get_chains():
            if chain.id == row["receptor_chain_id"]:
                protein = chain
            elif chain.id == row["ligand_chain_id"]:
                ligand = chain
        sample = self.featurizer.featurize(
            {"ligand": ligand, "protein": protein}
        )
        sample["target"] = row["label"]
        return sample

    @property
    def pos_weight(self) -> torch.Tensor:
        """To compute the weight of the positive class, assuming binary
        classification"""
        class_sizes = self.meta_df["label"].value_counts()
        pos_weights = np.mean(class_sizes) / class_sizes
        pos_weights = torch.from_numpy(pos_weights.values.astype(np.float32))
        return pos_weights[1] / pos_weights[0]

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        protein_graphs = []
        protein_smiles_strings = []
        ligand_graphs = []
        ligand_smiles_strings = []
        complex_graphs = []
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            g_targets.append(rec["target"])
            if "protein_smiles_strings" in rec:
                protein_smiles_strings.extend(rec["protein_smiles_strings"])
            if "ligand_smiles_strings" in rec:
                ligand_smiles_strings.extend(rec["ligand_smiles_strings"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "g_targets": torch.tensor(g_targets)
            .to(torch.float32)
            .unsqueeze(-1),
            "protein_smiles_strings": protein_smiles_strings,
            "ligand_smiles_strings": ligand_smiles_strings,
            "ligand_smiles": None,
        }
