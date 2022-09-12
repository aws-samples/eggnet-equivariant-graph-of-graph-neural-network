"""
Pytorch dataset classes from PPI prediction.
"""
from rdkit import Chem
import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch
import torch.utils.data as data
import torch.nn.functional as F
import dgl
from tqdm import tqdm

# custom modules
from ppi.data_utils import remove_nan_residues, mol_to_pdb_structure


class BasePPIDataset(data.Dataset):
    """Dataset for the Base Protein Graph."""

    def __init__(self, data_list, preprocess=False):
        super(BasePPIDataset, self).__init__()

        self.data_list = data_list
        if preprocess:
            print("Preprocessing data...")
            self._preprocess_all()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if isinstance(self.data_list[i], dict):
            # if not processed, process this instance and update
            self.data_list[i] = self._preprocess(self.data_list[i])
        return self.data_list[i]

    def _preprocess(self, complex):
        raise NotImplementedError

    def _preprocess_all(self):
        """Preprocess all the records in `data_list` with `_preprocess"""
        for i in tqdm(range(len(self.data_list))):
            self.data_list[i] = self._preprocess(self.data_list[i])


def prepare_pepbdb_data_list(parsed_structs: dict, df: pd.DataFrame) -> list:
    """
    Prepare the data_list required to construct PepBDBData object.
    Returns:
        - a list of protein complex objects/dict:
            {
                'pdb_id': str,
                'chain_id1': str,
                'chain_id2': str,
                'protein1': {'seq': str, 'coords': list[list[int]], 'name': str},
                'protein2': {'seq': str, 'coords': list[list[int]], 'name': str}
            }
    """
    data_list = []

    for pdb_id, rec in parsed_structs.items():
        sub_df = df.loc[df["PDB ID"] == pdb_id]

        for i, row in sub_df.iterrows():
            chain_id1, chain_id2 = (
                row["protein chain ID"],
                row["peptide chain ID"],
            )
            if chain_id1 not in rec or chain_id2 not in rec:
                # one of the chain doesn't have valid structure in PDB
                continue
            protein_complex = {
                "pdb_id": pdb_id,
                "chain_id1": chain_id1,
                "chain_id2": chain_id2,
            }
            # remove residues with nan's in coords
            rec[chain_id1] = remove_nan_residues(rec[chain_id1])
            rec[chain_id2] = remove_nan_residues(rec[chain_id2])
            if rec[chain_id1] and rec[chain_id2]:
                if (
                    len(rec[chain_id1]["seq"]) > 0
                    and len(rec[chain_id2]["seq"]) > 0
                ):
                    # both chains need to have residues with coords available
                    protein_complex["protein1"] = rec[chain_id1]
                    protein_complex["protein2"] = rec[chain_id2]
                    data_list.append(protein_complex)

    return data_list


class PepBDBDataset(BasePPIDataset):
    """
    Dataset representing PepBDB from http://huanglab.phys.hust.edu.cn/pepbdb/db/download/
    Each entry contains a pair of polypeptide chains from a PDB complex.
    """

    def __init__(
        self,
        data_list,
        contact_threshold=7.5,
        featurizer1=None,
        featurizer2=None,
        **kwargs
    ):
        self.contact_threshold = contact_threshold
        self.featurizer1 = featurizer1
        self.featurizer2 = featurizer2
        super(PepBDBDataset, self).__init__(data_list, **kwargs)

    def _preprocess(self, parsed_structure: dict):
        coords_1 = np.asarray(parsed_structure["protein1"]["coords"])
        coords_2 = np.asarray(parsed_structure["protein2"]["coords"])
        # CA-CA distance:
        contact_map = pairwise_distances(
            coords_1[:, 1], coords_2[:, 1], metric="euclidean"
        )
        y = contact_map < self.contact_threshold
        g1 = self.featurizer1.featurize(parsed_structure["protein1"])
        g2 = self.featurizer2.featurize(parsed_structure["protein2"])
        return (g1, g2), y

    def collate_fn(self, samples):
        g1s, g2s = [], []
        for (g1, g2), y in samples:
            g1s.append(g1)
            g2s.append(g2)
            y = y.sum(axis=1) > 0  # interacting residues on protein1
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
            # shape: (n_nodes, 1)
            n_nodes = g1.num_nodes()
            assert y.shape[0] == n_nodes
            g1.ndata["target"] = y
        return dgl.batch(g1s), dgl.batch(g2s)


class PepBDBComplexDataset(BasePPIDataset):
    """
    Dataset representing PepBDB from http://huanglab.phys.hust.edu.cn/pepbdb/db/download/
    Each entry contains a pair of polypeptide chains from a PDB complex.
    This class featurize the protein complex as a whole.
    """

    def __init__(
        self, data_list, contact_threshold=7.5, featurizer=None, **kwargs
    ):
        self.contact_threshold = contact_threshold
        self.featurizer = featurizer
        super(PepBDBComplexDataset, self).__init__(data_list, **kwargs)

    def _preprocess(self, parsed_structure: dict):
        coords_1 = np.asarray(parsed_structure["protein1"]["coords"])
        coords_2 = np.asarray(parsed_structure["protein2"]["coords"])
        # CA-CA distance:
        contact_map = pairwise_distances(
            coords_1[:, 1], coords_2[:, 1], metric="euclidean"
        )
        y = contact_map < self.contact_threshold
        g = self.featurizer.featurize(parsed_structure)
        return g, y

    def collate_fn(self, samples):
        """Collating protein complex graphs."""
        graphs = []
        for g, y in samples:
            # interacting residues on protein1
            y1 = torch.tensor(y.sum(axis=1) > 0, dtype=torch.float32)
            # interacting residues on protein2
            y2 = torch.tensor(y.sum(axis=0) > 0, dtype=torch.float32)
            y = torch.cat((y1, y2)).unsqueeze(-1)  # shape: (n_nodes, 1)
            n_nodes = g.num_nodes()
            assert y.shape[0] == n_nodes
            g.ndata["target"] = y

            graphs.append(g)
        return dgl.batch(graphs)


class ComplexBatch(dict):
    """
    A custom batch enabling memory pinning.
    ref: https://pytorch.org/docs/stable/data.html#memory-pinning
    """

    def __init__(self, graphs: dgl.DGLHeteroGraph, g_targets: torch.Tensor):
        self["graph"] = graphs
        self["g_targets"] = g_targets

    # custom memory pinning method on custom type
    def pin_memory(self):
        self["graph"].pin_memory_()  # TODO: this doesn't pin yet
        self["g_targets"] = self["g_targets"].pin_memory()
        return self


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
    ):
        self.keys = np.array(keys).astype(np.unicode_)
        self.data_dir = data_dir
        self.id_to_y = pd.Series(id_to_y, dtype=np.float32)
        self.featurizer = featurizer
        self.processed_data = pd.Series([None] * len(self))

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
            m2 = mol_to_pdb_structure(m2)

        graph, smiles_strings = self.featurizer.featurize(
            {
                "ligand": m1,
                "protein": m2,
            }
        )
        sample = {"graph": graph, "smiles_strings": smiles_strings}
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        graphs = []
        smiles_strings = []
        g_targets = []
        for rec in samples:
            graphs.append(rec["graph"])
            g_targets.append(rec["affinity"])
            smiles_strings.extend(rec["smiles_strings"])
        return {
            "graph": dgl.batch(graphs),
            "g_targets": torch.tensor(g_targets)
            .to(torch.float32)
            .unsqueeze(-1),
            "smiles_strings": smiles_strings,
        }
