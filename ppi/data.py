"""
Pytorch dataset classes from PPI prediction.
"""
from ppi.data_utils.pignet_featurizers import mol_to_feature
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

import numpy as np
import pickle

# custom modules
from ppi.data_utils import (
    remove_nan_residues,
    mol_to_pdb_structure,
    residue_to_mol,
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
        max_dims = max_tensor.shape
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
        if batch[j] == None:
            continue
        keys = []
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value

    return ret_dict


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
        compute_energy=False,
    ):
        self.keys = np.array(keys).astype(np.unicode_)
        self.data_dir = data_dir
        self.id_to_y = pd.Series(id_to_y)
        self.featurizer = featurizer
        self.processed_data = pd.Series([None] * len(self))
        self.compute_energy = compute_energy

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
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        if self.compute_energy:
            if protein_mol is None:
                protein_mol = residue_to_mol(protein_pdb)
            physics = mol_to_feature(m1, protein_mol)
            sample["physics"] = physics
            # a boolean mask to indicate nodes from proteins:
            mask = torch.zeros(sample["graph"].num_nodes())
            n_nodes_protein = physics["target_valid"].shape[0]
            mask[:n_nodes_protein] = 1
            sample["graph"].ndata["mask"] = mask
        return sample

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
        return {
            "graph": dgl.batch(graphs),
            "g_targets": torch.tensor(g_targets)
            .to(torch.float32)
            .unsqueeze(-1),
            "sample": tensor_collate_fn(physics),
            "smiles_strings": smiles_strings,
        }


def get_mask(protein_seq, pad_seq_len):
    """
    Used by CAMP dataset
    """
    if len(protein_seq) <= pad_seq_len:
        a = np.zeros(pad_seq_len)
        a[: len(protein_seq)] = 1
    else:
        cut_protein_seq = protein_seq[:pad_seq_len]
        a = np.zeros(pad_seq_len)
        a[: len(cut_protein_seq)] = 1
    return a


def boost_mask_BCE_loss(input_mask, flag):
    """
    Used by CAMP dataset ; flag is an indicator for checking whether this record has binding sites information
    """

    def conditional_BCE(y_true, y_pred):
        loss = flag * K.binary_crossentropy(y_true, y_pred) * input_mask
        return K.sum(loss) / K.sum(input_mask)

    return conditional_BCE


def prepare_camp_data_arrays(datafile, data_dir):
    """
    Input:
        datafile: peptide and protein sequence file with columns protein_seq, peptide_seq, protein_ss, peptide_ss
           (generated by Step 4 script as test_data.tsv).
        data_dir: Directory with subfolder dense_feature dict, that contains dictionaries of necessary features.
            You must have run Steps 1-5 to generate the feature dictionaries for the protein and peptides in subfolder dense_feature_dict
    Output:

    """
    print("loading features:")
    with open(
        data_dir + "/dense_feature_dict/protein_feature_dict", "rb"
    ) as f:
        protein_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/peptide_feature_dict", "rb"
    ) as f:
        peptide_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/protein_ss_feature_dict", "rb"
    ) as f:
        protein_ss_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/peptide_ss_feature_dict", "rb"
    ) as f:
        peptide_ss_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/protein_2_feature_dict", "rb"
    ) as f:
        protein_2_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/peptide_2_feature_dict", "rb"
    ) as f:
        peptide_2_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/protein_dense_feature_dict", "rb"
    ) as f:
        protein_dense_feature_dict = pickle.load(f, encoding="latin1")
    with open(
        data_dir + "/dense_feature_dict/peptide_dense_feature_dict", "rb"
    ) as f:
        peptide_dense_feature_dict = pickle.load(f, encoding="latin1")
    X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p = [], [], [], [], [], []
    X_dense_pep, X_dense_p = [], []
    pep_sequence, prot_sequence, Y = [], [], []
    X_pep_mask, X_bs_flag = [], []

    with open(datafile) as f:
        for line in f.readlines()[1:]:
            seq, peptide, peptide_ss, seq_ss = line.strip().split("\t")
            flag = 1.0  # For prediction, set flag==1 to generate binding sites
            X_pep_mask.append(get_mask(peptide, pad_pep_len))
            X_bs_flag.append(flag)

            pep_sequence.append(peptide)
            prot_sequence.append(seq)
            X_pep.append(peptide_feature_dict[peptide])
            X_p.append(protein_feature_dict[seq])
            X_SS_pep.append(peptide_ss_feature_dict[peptide_ss])
            X_SS_p.append(protein_ss_feature_dict[seq_ss])
            X_2_pep.append(peptide_2_feature_dict[peptide])
            X_2_p.append(protein_2_feature_dict[seq])
            X_dense_pep.append(peptide_dense_feature_dict[peptide])
            X_dense_p.append(protein_dense_feature_dict[seq])

    X_pep = np.array(X_pep)
    X_p = np.array(X_p)
    X_SS_pep = np.array(X_SS_pep)
    X_SS_p = np.array(X_SS_p)
    X_2_pep = np.array(X_2_pep)
    X_2_p = np.array(X_2_p)
    X_dense_pep = np.array(X_dense_pep)
    X_dense_p = np.array(X_dense_p)

    X_pep_mask = np.array(X_pep_mask)
    X_bs_flag = np.array(X_bs_flag)

    pep_sequence = np.array(pep_sequence)
    prot_sequence = np.array(prot_sequence)
    return (
        X_pep,
        X_p,
        X_SS_pep,
        X_SS_p,
        X_2_pep,
        X_2_p,
        X_dense_pep,
        X_dense_p,
        pep_sequence,
        prot_sequence,
        X_pep_mask,
        X_bs_flag,
    )


class CAMPDataset(data.Dataset):
    """
    Dataset required for CAMP predictor.
    Input:
        datafile: peptide and protein sequence file with columns protein_seq, peptide_seq, protein_ss, peptide_ss
           (generated by Step 4 script as test_data.tsv).
        data_dir: Directory with subfolder dense_feature dict, that contains dictionaries of necessary features.
            You must have run Steps 1-5 to generate the feature dictionaries for the protein and peptides in subfolder dense_feature_dict

    """

    def __init__(self, datafile, data_dir, model_mode=1):
        self.datafile = datafile
        self.data_dir = data_dir
        self.model_mode = model_mode
        self.FlagFeatures = False

    def load_features(self):
        (
            X_pep,
            X_p,
            X_SS_pep,
            X_SS_p,
            X_2_pep,
            X_2_p,
            X_dense_pep,
            X_dense_p,
            pep_sequence,
            prot_sequence,
            X_pep_mask,
            X_bs_flag,
        ) = prepare_camp_data_arrays(
            datafile=self.datafile, data_dir=self.data_dir
        )
        self.X_pep = X_pep
        self.X_p = X_p
        self.X_SS_pep = X_SS_pep
        self.X_SS_p = X_SS_p
        self.X_2_pep = X_2_pep
        self.X_2_p = X_2_p
        self.X_dense_pep = X_dense_pep
        self.X_dense_p = X_dense_p
        self.pep_sequence = prep_sequence
        self.prot_sequence = prot_sequence
        self.X_pep_mask = X_pep_mask
        self.X_bs_flag = X_bs_flag

        self.FlagFeatures = True

    def __len__(self) -> int:
        if self.FlagFeatures is False:
            self.load_features()
        return len(self.X_pep)

    def _preprocess(self, ix: int):
        # Return arrays may need to be re-specified as np.array(x)
        if self.FlagFeatures is False:
            self.load_features()
        if self.model_mode == 1:
            return (
                self.X_pep[ix],
                self.X_p[ix],
                self.X_SS_pep[ix],
                self.X_SS_p[ix],
                self.X_2_pep[ix],
                self.X_2_p[ix],
                self.X_dense_pep[ix],
                self.pep_sequence[ix],
                self.prot_sequence[ix],
            )
        else:
            return (
                self.X_pep[ix],
                self.X_p[ix],
                self.X_SS_pep[ix],
                self.X_SS_p[ix],
                self.X_2_pep[ix],
                self.X_2_p[ix],
                self.X_dense_pep[ix],
                self.pep_sequence[ix],
                self.prot_sequence[ix],
                self.X_pep_mask[ix],
                self.X_bs_flag[ix],
            )

    def __getitem__(self, idx: int):
        if self.processed_data[idx] is None:
            self.processed_data[idx] = self._preprocess(idx)
        return self.processed_data[idx]

    def collate_fn(self, samples):
        """
        Write this after testing out predict_CAMP.py with batch inference
        """


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

        protein_graph, ligand_graph, complex_graph = self.featurizer.featurize(
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
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        protein_graphs, ligand_graphs, complex_graphs = [], [], []
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            g_targets.append(rec["affinity"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "g_targets": torch.tensor(g_targets).unsqueeze(-1),
        }


class PIGNetAtomicBigraphComplexDataset(data.Dataset):
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

        protein_graph, ligand_graph, complex_graph = self.featurizer.featurize(
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
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        protein_graphs, ligand_graphs, complex_graphs = [], [], []
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            g_targets.append(rec["affinity"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "g_targets": torch.tensor(g_targets).unsqueeze(-1),
        }


class PIGNetAtomicBigraphComplexEnergyDataset(data.Dataset):
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

        (
            protein_graph,
            ligand_graph,
            complex_graph,
            physics,
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
            "sample": physics,
        }
        sample["affinity"] = self.id_to_y[key] * -1.36
        sample["key"] = key
        return sample

    def collate_fn(self, samples):
        """Collating protein complex graphs and graph-level targets."""
        protein_graphs, ligand_graphs, complex_graphs, physics = [], [], [], []
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            physics.append(rec["sample"])
            g_targets.append(rec["affinity"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "sample": tensor_collate_fn(physics),
            "g_targets": torch.tensor(g_targets).unsqueeze(-1),
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
        ) = ([], [], [], [], [])
        g_targets = []
        for rec in samples:
            protein_graphs.append(rec["protein_graph"])
            ligand_graphs.append(rec["ligand_graph"])
            complex_graphs.append(rec["complex_graph"])
            physics.append(rec["sample"])
            atom_to_residues.append(rec["atom_to_residue"])
            g_targets.append(rec["affinity"])
        return {
            "protein_graph": dgl.batch(protein_graphs),
            "ligand_graph": dgl.batch(ligand_graphs),
            "complex_graph": dgl.batch(complex_graphs),
            "sample": tensor_collate_fn(physics),
            "atom_to_residue": atom_to_residues,
            "g_targets": torch.tensor(g_targets).unsqueeze(-1),
        }
