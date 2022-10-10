"""
Utils for featurizing a small molecule (amino acid residue) from their structures.
"""
import dgl
from typing import Union, List
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import numpy as np
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from dgllife.model import load_pretrained
from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set


def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}


def featurize_bonds(mol):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'type': torch.tensor(feats).reshape(-1, 1).float()}


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
        fps = fps.ToBitString().encode()
        fps_vec = torch.from_numpy(np.frombuffer(fps, "u1") - ord("0"))
        return fps_vec


class GINFeaturizer(BaseResidueFeaturizer, nn.Module):
    """
    Convert a molecule to atom graph, then apply pretrained GNN
    to featurize the graph as a vector.
    """

    def __init__(self, gin_model, readout='attention', requires_grad=False, device="cpu"):
        nn.Module.__init__(self)
        BaseResidueFeaturizer.__init__(self)
        self.gin_model = gin_model
        self.device = device
        self.requires_grad = requires_grad

        emb_dim = gin_model.

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if gin_model.JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((self.gin_model.num_layers + 1) * self.gin_model.emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(self.gin_model.emb_dim, 1))
        elif readout == 'set2set':
            self.readout = Set2Set()
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max', 'attention' or 'set2set', got {}".format(readout))

    def _featurize(self, smiles: Union[str, List[str]]) -> torch.tensor:
        graphs = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            graph = mol_to_bigraph(mol, 
                                   node_featurizer=featurize_atoms,
                                   edge_featurizer=featurize_bonds)
            graphs.append(graph)
        g = dgl.batch(graphs)
        g = g.to(self.device)
        if not self.requires_grad:
            with torch.no_grad():
                node_feats = self.gin_model(g, g.ndata['atomic'].tolist(), g.edata['type'].tolist())
                graph_feats = self.readout(g, node_feats)
        else:
            node_feats = self.gin_model(g, g.ndata['atomic'].tolist(), g.edata['type'].tolist())
            graph_feats = self.readout(g, node_feats)
        return graph_feats

    def forward(self, smiles: str) -> torch.tensor:
        """Expose this method when we want to unfreeze the network,
        training jointly with higher level GNN"""
        assert self.requires_grad
        return self._featurize(smiles)

    @property
    def output_size(self) -> int:
        return self.gin_model.emb_dim

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
    gin_names = ('gin_supervised_contextpred', 
                'gin_supervised_infomax',
                'gin_supervised_edgepred',
                'gin_supervised_masking')
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
    elif name.lower().startswith("gin"):
        requires_grad = True if "grad" in name else False
        name_split = name.split("-")
        readout = name_split[3]
        name = "_".join(name_split[0:3])
        name = name.lower()
        print(name)
        assert name in gin_names
        gin_model = load_pretrained(name)
        print(gin_model)
        residue_featurizer = GINFeaturizer(gin_model=gin_model, readout=readout, requires_grad=requires_grad)
    else:
        raise NotImplementedError
    return residue_featurizer
