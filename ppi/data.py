"""
Pytorch dataset classes from PPI prediction.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torch.utils.data as data
import torch.nn.functional as F

# custom modules
from ppi import data_utils


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
