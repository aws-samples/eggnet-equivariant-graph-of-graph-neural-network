"""
Pytorch dataset classes from PPI prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch
import torch.utils.data as data
import torch.nn.functional as F
import dgl

# custom modules
from ppi.data_utils import remove_nan_residues


class BasePPIDataset(data.Dataset):
    """Dataset for the Base Protein Graph."""

    def __init__(self, data_list):
        super(BasePPIDataset, self).__init__()

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._preprocess(self.data_list[i])

    def _preprocess(self, complex):
        raise NotImplementedError


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
            y = torch.tensor(y, dtype=torch.float32)  # shape: (n_nodes, )
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
            y = torch.cat((y1, y2))  # shape: (n_nodes, )
            n_nodes = g.num_nodes()
            assert y.shape[0] == n_nodes
            g.ndata["target"] = y

            graphs.append(g)
        return dgl.batch(graphs)
