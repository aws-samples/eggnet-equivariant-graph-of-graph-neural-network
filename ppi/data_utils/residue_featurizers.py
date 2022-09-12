"""
Utils for featurizing a small molecule (amino acid residue) from their structures.
"""
from typing import Union, List
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import numpy as np
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem


class BaseResidueFeaturizer(object):
    """A simple base class with caching"""

    def __init__(self):
        self.cache = {}

    def featurize(self, smiles: str) -> torch.tensor:
        if smiles not in self.cache:
            self.cache[smiles] = self._featurize(smiles)
        return self.cache[smiles]

    def _featurize(self, smiles: str) -> torch.tensor:
        raise NotImplementedError


class FingerprintFeaturizer(BaseResidueFeaturizer):
    """
    https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-fingerprints
    """

    def __init__(self, fingerprint_type):
        self.fingerprint_type = fingerprint_type
        super(FingerprintFeaturizer, self).__init__()

    def _featurize(self, smiles: str) -> torch.tensor:
        mol = Chem.MolFromSmiles(smiles)
        if self.fingerprint_type == "MACCS":
            fps = MACCSkeys.GenMACCSKeys(mol)
        elif self.fingerprint_type == "Morgan":
            fps = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, useFeatures=True, nBits=1024
            )
        else:
            raise NotImplementedError
        # convert ExplicitBitVect to uint vector:
        fps_vec = torch.frombuffer(
            fps.ToBitString().encode(), dtype=torch.uint8
        ) - ord("0")
        return fps_vec


class GNNFeaturizer(BaseResidueFeaturizer):
    """
    Convert a molecule to atom graph, then apply pretrained GNN
    to featurize the graph as a vector.
    """

    def __init__(self, gnn_model, device="cpu"):
        self.gnn_model = gnn_model
        self.device = device
        super(GNNFeaturizer, self).__init__()

    def _featurize(self, smiles: str) -> torch.tensor:
        mol = Chem.MolFromSmiles(smiles)
        g = mol_to_bigraph(mol)
        g = g.to(self.device)
        return self.gnn_model(g)


class MolT5Featurizer(BaseResidueFeaturizer, nn.Module):
    """
    Use MolT5 encodings as residue features.
    """

    def __init__(
        self,
        model_size="small",
        model_max_length=512,
        requires_grad=False,
    ):
        """
        Args:
            model_size: one of ('small', 'base', 'large')
        """
        nn.Module.__init__(self)
        BaseResidueFeaturizer.__init__(self)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "laituan245/molt5-%s" % model_size,
            model_max_length=model_max_length,
        )
        self.model = T5EncoderModel.from_pretrained(
            "laituan245/molt5-%s" % model_size
        )
        self.requires_grad = requires_grad

    def _featurize(self, smiles: Union[str, List[str]]) -> torch.tensor:
        input_ids = self.tokenizer(
            smiles, return_tensors="pt", padding=True
        ).input_ids
        input_ids = input_ids.to(self.model.device)
        if not self.requires_grad:
            with torch.no_grad():
                outputs = self.model(input_ids)
        else:
            outputs = self.model(input_ids)

        # n_smiles_strings = 1 if type(smiles) is str else len(smiles)
        # shape: [n_smiles_strings, input_ids.shape[1], model_max_length]
        last_hidden_states = outputs.last_hidden_state

        # average over positions:
        return last_hidden_states.mean(axis=1).squeeze(0)

    def forward(self, smiles: str) -> torch.tensor:
        """Expose this method when we want to unfreeze the network,
        training jointly with higher level GNN"""
        assert self.requires_grad
        return self._featurize(smiles)

    @property
    def output_size(self) -> int:
        return self.model.config.d_model


def get_residue_featurizer(name=""):
    """
    Handles initializing the residue featurizer.
    """
    fingerprint_names = ("MACCS", "Morgan")
    if name in fingerprint_names:
        residue_featurizer = FingerprintFeaturizer(name)
    elif name.lower().startswith("molt5"):
        model_size = "small"
        if "-" in name:
            model_size = name.split("-")[1]
        requires_grad = True if "grad" in name else False
        residue_featurizer = MolT5Featurizer(
            model_size=model_size, requires_grad=requires_grad
        )
    return residue_featurizer
