"""
Utils for featurizing a small molecule (amino acid residue) from their structures.
"""
import torch
import numpy as np
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem


class FingerprintFeaturizer(object):
    """
    https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-fingerprints
    """

    def __init__(self, fingerprint_type):
        self.fingerprint_type = fingerprint_type

    def featurize(self, smiles: str) -> torch.tensor:
        mol = Chem.MolFromSmiles(smiles)
        if self.fingerprint_type == "MACCS":
            fps = MACCSkeys.GenMACCSKeys(mol)
        elif self.fingerprint_type == "Morgan":
            fps = AllChem.GetMorganFingerprint(
                mol, 2, useFeatures=True, nBits=1024
            )
        else:
            raise NotImplementedError
        # convert ExplicitBitVect to uint vector:
        fps_vec = torch.frombuffer(
            fps.ToBitString().encode(), dtype=torch.uint8
        ) - ord("0")
        return fps_vec


class GNNFeaturizer(object):
    """
    Convert a molecule to atom graph, then apply pretrained GNN
    to featurize the graph as a vector.
    """

    def __init__(self, gnn_model, device):
        self.gnn_model = gnn_model
        self.device = device

    def featurize(self, smiles: str) -> torch.tensor:
        mol = Chem.MolFromSmiles(smiles)
        g = mol_to_bigraph(mol)
        g = g.to(self.device)
        return self.gnn_model(g)
