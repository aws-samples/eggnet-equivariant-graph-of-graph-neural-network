"""
Utils for featurizing a polypeptide (protein or peptide) from their structures.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import dgl

from rdkit import Chem
from Bio.PDB.Polypeptide import is_aa
from . import contact_map_utils as utils

from dgllife.utils import mol_to_bigraph
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer

from .pignet_featurizers import mol_to_feature


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.

    Args:
        tensor: Torch tensor to be normalized.
        dim: Integer. Dimension to normalize across.

    Returns:
        Normalized tensor with zeros instead of nan's or infinity values.

    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].


    Args:
        D: generic torch tensor
        D_min: Float. Minimum of the sequence of numbers created.
        D_max: Float. Max of the sequence of numbers created.
        D_count: Positive integer. Count of the numbers in the sequence. It is also lenght of the new dimension (-1) created in D.
        device: Device where D is stored.

    Return:
        Input `D` matrix with an RBF embedding along axis -1.
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


class BaseFeaturizer(object):
    """Base class for polypeptide and complex featurizer with common feature extractions:
    - dihedral angles on nodes
    - positional embeddings for edges
    - atom pair orientations
    - side chain directions on nodes
    """

    def __init__(
        self, num_positional_embeddings=16, top_k=30, num_rbf=16, device="cpu"
    ):
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device

    def featurize(self, item: dict) -> dgl.DGLGraph:
        raise NotImplementedError

    def _dihedrals(self, X, eps=1e-7):
        """Compute sines and cosines dihedral angles (phi, psi, and omega)

        Args:
            X: torch.Tensor specifying coordinates of key atoms (N, CA, C, O) in 3D space with shape [seq_len, 4, 3]
            eps: Float defining the epsilon using to clamp the angle between normals: min= -1*eps, max=1-eps

        Returns:
            Sines and cosines dihedral angles as a torch.Tensor of shape [seq_len, 6]
        """
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index, num_embeddings=None):
        """Creates and returns the positional embeddings.

        Args:
            edge_index: torch.Tensor representing edges in COO format with shape [2, num_edges].
            num_embeddings: Integer representing the number of embeddings.

        Returns:
            Positional embeddings as a torch tensor
        """
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(
                0, num_embeddings, 2, dtype=torch.float32, device=self.device
            )
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        """Compute orientations between pairs of atoms from neighboring residues.

        Args:
            X: torch.Tensor representing atom coordinates with shape [n_atoms, 3]

        Returns:
            torch.Tensor atom pair orientations
        """
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        """Compute the unit vector representing the imputed side chain directions (C_beta - C_alpha).

        Args:
            X: torch.Tensor specifying coordinates of key atoms (N, CA, C, O) in 3D space with shape [seq_len, 4, 3]

        Returns:
            Torch tensor representing side chain directions with shape [seq_len, 3]
        """
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


class NaturalPolypeptideFeaturizer(BaseFeaturizer):
    """
    For individual polypeptide chain that only contains 20 natural amino acids (can work
    with unknown residues 'X')
    """

    def __init__(self, **kwargs):

        self.letter_to_num = {
            "C": 4,
            "D": 3,
            "S": 15,
            "Q": 5,
            "K": 11,
            "I": 9,
            "P": 14,
            "T": 16,
            "F": 13,
            "A": 0,
            "G": 7,
            "H": 8,
            "E": 6,
            "L": 10,
            "R": 1,
            "W": 17,
            "V": 19,
            "N": 2,
            "Y": 18,
            "M": 12,
            "X": 0,
        }
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        super(NaturalPolypeptideFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein: dict) -> dgl.DGLGraph:
        """Featurizes the protein information as a graph for the GNN

        Args:
            protein: Dictionary with the protein seq, coord and name.

        Returns:
            dgl.graph instance representing with the protein information
        """
        name = protein.get("name")
        with torch.no_grad():
            coords = torch.as_tensor(
                protein["coords"], device=self.device, dtype=torch.float32
            )
            seq = torch.as_tensor(
                [self.letter_to_num[a] for a in protein["seq"]],
                device=self.device,
                dtype=torch.long,
            )

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            # construct knn graph from C-alpha coordinates
            g = dgl.knn_graph(X_ca, k=min(self.top_k, X_ca.shape[0]))
            edge_index = g.edges()

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(
                E_vectors.norm(dim=-1),
                D_count=self.num_rbf,
                device=self.device,
            )

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

        # node features
        g.ndata["node_s"] = node_s
        g.ndata["node_v"] = node_v
        g.ndata["mask"] = mask
        g.ndata["seq"] = seq  # one-hot encoded amino acid sequence
        # edge features
        g.edata["edge_s"] = edge_s
        g.edata["edge_v"] = edge_v
        # graph attrs
        setattr(g, "name", name)
        return g


class NaturalComplexFeaturizer(NaturalPolypeptideFeaturizer):
    """
    For protein complex (multiple polypeptide chains) that only contains 20
    natural amino acids (can work with unknown residues 'X')
    """

    def __init__(self, **kwargs):
        super(NaturalComplexFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dgl.DGLGraph:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'pdb_id': str,
                'chain_id1': str,
                'chain_id2': str,
                'protein1': {'seq': str, 'coords': list[list[int]], 'name': str},
                'protein2': {'seq': str, 'coords': list[list[int]], 'name': str}
            }
        Returns:
            dgl.graph instance representing with the protein complex information
        """
        protein1 = protein_complex["protein1"]
        protein2 = protein_complex["protein2"]
        with torch.no_grad():
            coords = torch.as_tensor(
                protein1["coords"] + protein2["coords"],
                device=self.device,
                dtype=torch.float32,
            )  # shape: [seq_len1 + seq_len2, 4, 3]

            seq = torch.as_tensor(
                [
                    self.letter_to_num[a]
                    for a in protein1["seq"] + protein2["seq"]
                ],
                device=self.device,
                dtype=torch.long,
            )  # shape: [seq_len1 + seq_len2]

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            # construct knn graph from C-alpha coordinates
            g = dgl.knn_graph(X_ca, k=min(self.top_k, X_ca.shape[0]))
            edge_index = g.edges()

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(
                E_vectors.norm(dim=-1),
                D_count=self.num_rbf,
                device=self.device,
            )

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

        # node features
        g.ndata["node_s"] = node_s
        g.ndata["node_v"] = node_v
        g.ndata["mask"] = mask
        g.ndata["seq"] = seq  # one-hot encoded amino acid sequences
        # edge features
        g.edata["edge_s"] = edge_s
        g.edata["edge_v"] = edge_v
        # graph attrs
        # setattr(g, "name", protein_complex["name"])
        return g


class NoncanonicalComplexFeaturizer(BaseFeaturizer):
    """
    For protein complex (multiple polypeptide chains) that contains
    non-canonical and/or natural amino acids.

    The amino acid residues need to be represented as a list of smile strings.

    Args:
        - residue_featurizer: a function mapping a smile string to a vector
    """

    def __init__(self, residue_featurizer, **kwargs):
        self.residue_featurizer = residue_featurizer
        super(NoncanonicalComplexFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dgl.DGLGraph:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'pdb_id': str,
                'chain_id1': str,
                'chain_id2': str,
                'protein1': {'residues': list[str], 'coords': list[list[int]], 'name': str},
                'protein2': {'residues': list[str], 'coords': list[list[int]], 'name': str}
            }
        Returns:
            dgl.graph instance representing with the protein complex information
        """
        protein1 = protein_complex["protein1"]
        protein2 = protein_complex["protein2"]
        with torch.no_grad():
            coords = torch.as_tensor(
                protein1["coords"] + protein2["coords"],
                device=self.device,
                dtype=torch.float32,
            )  # shape: [seq_len1 + seq_len2, 4, 3]

            residues = torch.as_tensor(
                [
                    self.residue_featurizer(smiles)
                    for smiles in protein1["residues"] + protein2["residues"]
                ],
                device=self.device,
                dtype=torch.long,
            )  # shape: [seq_len1 + seq_len2, d_embed]

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            # construct knn graph from C-alpha coordinates
            g = dgl.knn_graph(X_ca, k=min(self.top_k, X_ca.shape[0]))
            edge_index = g.edges()

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(
                E_vectors.norm(dim=-1),
                D_count=self.num_rbf,
                device=self.device,
            )

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = torch.cat([dihedrals, residues], dim=-1)
            node_v = torch.cat(
                [orientations, sidechains.unsqueeze(-2)], dim=-2
            )
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

        # node features
        g.ndata["node_s"] = node_s
        g.ndata["node_v"] = node_v
        g.ndata["mask"] = mask
        # edge features
        g.edata["edge_s"] = edge_s
        g.edata["edge_v"] = edge_v
        # graph attrs
        # setattr(g, "name", protein_complex["name"])
        return g


class PDBBindComplexFeaturizer(BaseFeaturizer):
    """
    For protein complex from PDBBind dataset that contains multiple
    protein chains with natural amino acids and one ligand (small molecule).

    The amino acid residues need to be represented as a list of smile strings.

    Args:
        - residue_featurizer: a function mapping a smile string to a vector
    """

    def __init__(self, residue_featurizer, **kwargs):
        self.residue_featurizer = residue_featurizer
        super(PDBBindComplexFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dict:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'ligand': rdkit.Chem object of the ligand,
                'protein': PDB.Structure object of the protein,
            }
        Returns:
            if residue_featurizer is provided:
                dgl.graph instance representing with the protein complex information
            else:
                (dgl.graph, list_of_node_smiles_strings)
        """
        ligand, protein = protein_complex["ligand"], protein_complex["protein"]
        ligand = Chem.RemoveHs(ligand)

        protein_coords = []
        residue_smiles = []  # SMILES strings of residues in the protein
        for res in protein.get_residues():
            if is_aa(res):
                protein_coords.append(utils.get_atom_coords(res))
                res_mol = utils.residue_to_mol(res)
                residue_smiles.append(Chem.MolToSmiles(res_mol))

        # backbone ["N", "CA", "C", "O"] coordinates for proteins
        # shape: [seq_len, 4, 3]
        protein_coords = torch.as_tensor(
            np.asarray(protein_coords), dtype=torch.float32
        )

        # shape: [ligand_n_atoms, 3]
        ligand_coords = torch.as_tensor(
            ligand.GetConformers()[0].GetPositions(), dtype=torch.float32
        )
        # take the centroid of ligand atoms
        ligand_coords = ligand_coords.mean(axis=0).reshape(-1, 3)

        # combine protein and ligand coordinates
        X_ca = torch.cat((protein_coords[:, 1], ligand_coords), axis=0)

        # SMILES strings of AA residues and ligand
        # in the same order with the nodes in the graph
        smiles_strings = residue_smiles + [Chem.MolToSmiles(ligand)]
        if self.residue_featurizer:
            residues = (
                torch.stack(
                    [
                        self.residue_featurizer.featurize(smiles)
                        for smiles in smiles_strings
                    ]
                )
                .to(self.device)
                .to(torch.long)
            )
            # shape: [seq_len + 1, d_embed]

        # construct knn graph from C-alpha coordinates
        g = dgl.knn_graph(X_ca, k=min(self.top_k, X_ca.shape[0]))
        edge_index = g.edges()

        pos_embeddings = self._positional_embeddings(edge_index)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(
            E_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        dihedrals = self._dihedrals(protein_coords)
        orientations = self._orientations(X_ca)
        sidechains = self._sidechains(protein_coords)

        # dummy-fill for the ligand node
        dihedrals = torch.cat([dihedrals, torch.zeros(1, 6)])
        sidechains = torch.cat([sidechains, torch.zeros(1, 3)])

        if self.residue_featurizer:
            node_s = torch.cat([dihedrals, residues], dim=-1)
        else:
            node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        g.ndata["node_s"] = node_s.contiguous()
        g.ndata["node_v"] = node_v.contiguous()
        # edge features
        g.edata["edge_s"] = edge_s.contiguous()
        g.edata["edge_v"] = edge_v.contiguous()
        if self.residue_featurizer:
            return {"graph": g}
        else:
            return {"graph": g, "smiles_strings": smiles_strings}


class PIGNetHeteroBigraphComplexFeaturizer(BaseFeaturizer):
    """
    For protein complex from PDBBind dataset that contains multiple
    protein chains with natural amino acids and one ligand (small molecule).

    The amino acid residues need to be represented as a list of smile strings.

    Args:
        residue_featurizer: a function mapping a smile string to a vector

    Returns:
        (g_rec, g_lig): featurized receptor graph, featurized ligand graph
    """

    def __init__(self, residue_featurizer, molecular_featurizers="canonical", **kwargs):
        self.residue_featurizer = residue_featurizer
        if molecular_featurizers == "canonical":
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer()
            self.add_self_loop = False
        elif molecular_featurizers == "pretrained":
            self.node_featurizer = PretrainAtomFeaturizer()
            self.edge_featurizer = PretrainBondFeaturizer()
            self.add_self_loop = True
        else:
            raise NotImplementedError()
        super(PIGNetHeteroBigraphComplexFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dgl.DGLGraph:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'ligand': rdkit.Chem object of the ligand,
                'protein': PDB.Structure object of the protein,
            }
        Returns:
            dgl.graph instance representing with the protein complex information
        """
        ligand, protein = protein_complex["ligand"], protein_complex["protein"]
        ligand = Chem.RemoveHs(ligand)
        ligand_graph = mol_to_bigraph(mol=ligand,
                                      node_featurizer=self.node_featurizer, 
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=self.add_self_loop)

        protein_coords = []
        residue_smiles = []  # SMILES strings of residues in the protein
        for res in protein.get_residues():
            if is_aa(res):
                protein_coords.append(utils.get_atom_coords(res))
                res_mol = utils.residue_to_mol(res)
                residue_smiles.append(Chem.MolToSmiles(res_mol))

        # backbone ["N", "CA", "C", "O"] coordinates for proteins
        # shape: [seq_len, 4, 3]
        protein_coords = torch.as_tensor(
            np.asarray(protein_coords), dtype=torch.float32
        )

        # SMILES strings of AA residues and ligand
        # in the same order with the nodes in the graph
        smiles_strings = residue_smiles
        if self.residue_featurizer:
            residues = (
                torch.stack(
                    [
                        self.residue_featurizer.featurize(smiles)
                        for smiles in smiles_strings
                    ]
                )
                .to(self.device)
                .to(torch.long)
            )
            # shape: [seq_len + 1, d_embed]
        
        # construct knn graph from C-alpha coordinates
        ca_coords = protein_coords[:, 1]
        protein_graph = dgl.knn_graph(ca_coords, k=min(self.top_k, ca_coords.shape[0]))
        ca_edge_index = protein_graph.edges()

        pos_embeddings = self._positional_embeddings(ca_edge_index)
        ca_vectors = ca_coords[ca_edge_index[0]] - ca_coords[ca_edge_index[1]]
        rbf = _rbf(
            ca_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        dihedrals = self._dihedrals(protein_coords)
        orientations = self._orientations(ca_coords)
        sidechains = self._sidechains(protein_coords)

        if self.residue_featurizer:
            node_s = torch.cat([dihedrals, residues], dim=-1)
        else:
            node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(ca_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        protein_graph.ndata["node_s"] = node_s
        protein_graph.ndata["node_v"] = node_v
        # edge features
        protein_graph.edata["edge_s"] = edge_s
        protein_graph.edata["edge_v"] = edge_v

        # shape: [ligand_n_atoms, 3]
        ligand_coords = torch.as_tensor(
            ligand.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        ligand_vectors = ligand_coords[ligand_graph.edges()[0].long()] - ligand_coords[ligand_graph.edges()[1].long()]
        # ligand node features
        ligand_graph.ndata["node_s"] = ligand_graph.ndata["h"]
        ligand_graph.ndata["node_v"] = ligand_coords.unsqueeze(-2)
        # ligand edge features
        ligand_graph.edata["edge_s"] = ligand_graph.edata["e"]
        ligand_graph.edata["edge_v"] = _normalize(ligand_vectors).unsqueeze(-2)

        # combine protein and ligand coordinates
        X_cat = torch.cat((protein_coords[:, 1], ligand_coords), axis=0)

        # construct knn graph from C-alpha coordinates
        complex_graph = dgl.knn_graph(X_cat, k=min(self.top_k, X_cat.shape[0]))
        edge_index = complex_graph.edges()

        E_vectors = X_cat[edge_index[0]] - X_cat[edge_index[1]]
        rbf = _rbf(
            E_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        # protein_feat_pad = F.pad(protein_graph.ndata['node_s'], (0, ligand_graph.ndata['node_s'].shape[-1]))
        # ligand_feat_pad = F.pad(ligand_graph.ndata['node_s'], (protein_graph.ndata['node_s'].shape[-1], 0))

        if self.residue_featurizer:
            node_s = torch.cat([dihedrals, residues], dim=-1)
        else:
            node_s = dihedrals
        node_v = X_cat.unsqueeze(-2)
        edge_s = rbf
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        complex_graph.ndata["node_s"] = node_s
        complex_graph.ndata["node_v"] = node_v
        # edge features
        complex_graph.edata["edge_s"] = edge_s
        complex_graph.edata["edge_v"] = edge_v
        if self.residue_featurizer:
            return protein_graph, ligand_graph, complex_graph
        else:
            return protein_graph, ligand_graph, complex_graph, smiles_strings


class PIGNetAtomicBigraphGeometricComplexFeaturizer(BaseFeaturizer):
    """
    For protein complex from PDBBind dataset that contains multiple
    protein chains with natural amino acids and one ligand (small molecule).

    The amino acid residues need to be represented as a list of smile strings.

    Args:
        residue_featurizer: a function mapping a smile string to a vector

    Returns:
        (g_rec, g_lig): featurized receptor graph, featurized ligand graph
    """

    def __init__(self, residue_featurizer, molecular_featurizers="canonical", return_physics=False, **kwargs):
        self.residue_featurizer = residue_featurizer
        self.return_physics = return_physics
        if molecular_featurizers == "canonical":
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer()
            self.add_self_loop = False
        elif molecular_featurizers == "pretrained":
            self.node_featurizer = PretrainAtomFeaturizer()
            self.edge_featurizer = PretrainBondFeaturizer()
            self.add_self_loop = True
        else:
            raise NotImplementedError()
        super(PIGNetAtomicBigraphGeometricComplexFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dgl.DGLGraph:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'ligand': rdkit.Chem object of the ligand,
                'protein': rdkit.Chem object object of the protein,
            }
        Returns:
            dgl.graph instance representing with the protein complex information
        """
        ligand, protein = protein_complex["ligand"], protein_complex["protein"]
        if self.return_physics:
            sample = mol_to_feature(ligand_mol=ligand, target_mol=protein)

        ligand = Chem.RemoveHs(ligand)
        ligand_graph = mol_to_bigraph(mol=ligand,
                                      node_featurizer=self.node_featurizer, 
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=self.add_self_loop)

        protein = Chem.RemoveHs(protein)
        protein_graph = mol_to_bigraph(mol=protein,
                                      node_featurizer=self.node_featurizer, 
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=self.add_self_loop)

        # shape: [protein_n_atoms, 3]
        # shape: [seq_len, 4, 3]
        protein_coords = torch.as_tensor(
            protein.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        # shape: [ligand_n_atoms, 3]
        ligand_coords = torch.as_tensor(
            ligand.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        protein_vectors = protein_coords[protein_graph.edges()[0].long()] - protein_coords[protein_graph.edges()[1].long()]
        # protein node features
        protein_graph.ndata["node_s"] = protein_graph.ndata["h"]
        protein_graph.ndata["node_v"] = protein_coords.unsqueeze(-2)
        # protein edge features
        protein_graph.edata["edge_s"] = protein_graph.edata["e"]
        protein_graph.edata["edge_v"] = _normalize(protein_vectors).unsqueeze(-2)

        ligand_vectors = ligand_coords[ligand_graph.edges()[0].long()] - ligand_coords[ligand_graph.edges()[1].long()]
        # ligand node features
        ligand_graph.ndata["node_s"] = ligand_graph.ndata["h"]
        ligand_graph.ndata["node_v"] = ligand_coords.unsqueeze(-2)
        # ligand edge features
        ligand_graph.edata["edge_s"] = ligand_graph.edata["e"]
        ligand_graph.edata["edge_v"] = _normalize(ligand_vectors).unsqueeze(-2)

        # combine protein and ligand coordinates
        X_ca = torch.cat((protein_coords, ligand_coords), axis=0)

        # construct knn graph from C-alpha coordinates
        complex_graph = dgl.knn_graph(X_ca, k=min(self.top_k, X_ca.shape[0]))
        edge_index = complex_graph.edges()

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(
            E_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        node_s = torch.cat([protein_graph.ndata['h'], ligand_graph.ndata['h']], dim=0)
        node_v = X_ca.unsqueeze(-2)
        edge_s = rbf
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        complex_graph.ndata["node_s"] = node_s
        complex_graph.ndata["node_v"] = node_v
        # edge features
        complex_graph.edata["edge_s"] = edge_s
        complex_graph.edata["edge_v"] = edge_v
        if self.return_physics:
            return protein_graph, ligand_graph, complex_graph, sample
        else:
            return protein_graph, ligand_graph, complex_graph


class PIGNetAtomicBigraphPhysicalComplexFeaturizer(BaseFeaturizer):
    """
    For protein complex from PDBBind dataset that contains multiple
    protein chains with natural amino acids and one ligand (small molecule).

    The amino acid residues need to be represented as a list of smile strings.

    Args:
        residue_featurizer: a function mapping a smile string to a vector

    Returns:
        (g_rec, g_lig): featurized receptor graph, featurized ligand graph
    """

    def __init__(self, residue_featurizer, molecular_featurizers="canonical", return_physics=False, **kwargs):
        self.residue_featurizer = residue_featurizer
        self.return_physics = return_physics
        if molecular_featurizers == "canonical":
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer()
            self.add_self_loop = False
        elif molecular_featurizers == "pretrained":
            self.node_featurizer = PretrainAtomFeaturizer()
            self.edge_featurizer = PretrainBondFeaturizer()
            self.add_self_loop = True
        else:
            raise NotImplementedError()
        super(PIGNetAtomicBigraphPhysicalComplexFeaturizer, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dgl.DGLGraph:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'ligand': rdkit.Chem object of the ligand,
                'protein': rdkit.Chem object of the protein,
            }
        Returns:
            dgl.graph instance representing with the protein complex information
        """
        ligand, protein = protein_complex["ligand"], protein_complex["protein"]
        sample = mol_to_feature(ligand_mol=ligand, target_mol=protein)

        ligand = Chem.RemoveHs(ligand)
        ligand_graph = mol_to_bigraph(mol=ligand,
                                      node_featurizer=self.node_featurizer, 
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=self.add_self_loop)

        protein = Chem.RemoveHs(protein)
        protein_graph = mol_to_bigraph(mol=protein,
                                      node_featurizer=self.node_featurizer, 
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=self.add_self_loop)

        # shape: [protein_n_atoms, 3]
        # shape: [seq_len, 4, 3]
        protein_coords = torch.as_tensor(
            protein.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        # shape: [ligand_n_atoms, 3]
        ligand_coords = torch.as_tensor(
            ligand.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        protein_vectors = protein_coords[protein_graph.edges()[0].long()] - protein_coords[protein_graph.edges()[1].long()]
        # protein node features
        protein_graph.ndata["node_s"] = protein_graph.ndata["h"]
        protein_graph.ndata["node_v"] = protein_coords.unsqueeze(-2)
        # protein edge features
        protein_graph.edata["edge_s"] = protein_graph.edata["e"]
        protein_graph.edata["edge_v"] = _normalize(protein_vectors).unsqueeze(-2)

        ligand_vectors = ligand_coords[ligand_graph.edges()[0].long()] - ligand_coords[ligand_graph.edges()[1].long()]
        # ligand node features
        ligand_graph.ndata["node_s"] = ligand_graph.ndata["h"]
        ligand_graph.ndata["node_v"] = ligand_coords.unsqueeze(-2)
        # ligand edge features
        ligand_graph.edata["edge_s"] = ligand_graph.edata["e"]
        ligand_graph.edata["edge_v"] = _normalize(ligand_vectors).unsqueeze(-2)

        # construct complex graph from rdkit computed interactions
        n_ligand = sample['interaction_indice'].shape[1]
        n_protein = sample['interaction_indice'].shape[2]
        interaction_indice_pad = np.pad(sample['interaction_indice'], 
                                [(0, 0), (n_protein, 0), (0, n_ligand)])
        interaction_indice_symm = np.maximum(interaction_indice_pad, np.transpose(interaction_indice_pad, (0, 2, 1)))
        src, dst = np.nonzero(interaction_indice_symm.max(axis=0) + np.eye(n_protein+n_ligand))
        complex_graph = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)))
        edge_index = complex_graph.edges()

        # combine protein and ligand coordinates
        X_ca = torch.cat((protein_coords, ligand_coords), axis=0)

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(
            E_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        node_s = 1.*(torch.cat([torch.from_numpy(sample['target_h']), 
                            torch.from_numpy(sample['ligand_h'])], dim=0)).float()
        node_v = torch.cat([torch.from_numpy(sample['target_pos']), 
                            torch.from_numpy(sample['ligand_pos'])], dim=0).unsqueeze(-2).float()
        edge_s = torch.from_numpy(interaction_indice_symm[:, src, dst]).T.float()
        edge_v = _normalize(E_vectors).unsqueeze(-2).float()

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        complex_graph.ndata["node_s"] = node_s
        complex_graph.ndata["node_v"] = node_v
        # edge features
        complex_graph.edata["edge_s"] = edge_s
        complex_graph.edata["edge_v"] = edge_v
        if self.return_physics:
            return protein_graph, ligand_graph, complex_graph, sample
        else:
            return protein_graph, ligand_graph, complex_graph


class PIGNetHeteroBigraphComplexFeaturizerForEnergyModel(BaseFeaturizer):
    """
    For protein complex from PDBBind dataset that contains multiple
    protein chains with natural amino acids and one ligand (small molecule).

    The amino acid residues need to be represented as a list of smile strings.

    Args:
        residue_featurizer: a function mapping a smile string to a vector

    Returns:
        (g_rec, g_lig): featurized receptor graph, featurized ligand graph
    """

    def __init__(self, residue_featurizer, molecular_featurizers="canonical", **kwargs):
        self.residue_featurizer = residue_featurizer
        if molecular_featurizers == "canonical":
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer()
            self.add_self_loop = False
        elif molecular_featurizers == "pretrained":
            self.node_featurizer = PretrainAtomFeaturizer()
            self.edge_featurizer = PretrainBondFeaturizer()
            self.add_self_loop = True
        else:
            raise NotImplementedError()
        super(PIGNetHeteroBigraphComplexFeaturizerForEnergyModel, self).__init__(**kwargs)

    def featurize(self, protein_complex: dict) -> dgl.DGLGraph:
        """Featurizes the protein complex information as a graph for the GNN

        Args:
            protein_complex: dict
            {
                'ligand': rdkit.Chem object of the ligand,
                'protein': PDB.Structure object of the protein,
            }
        Returns:
            dgl.graph instance representing with the protein complex information
        """
        ligand, protein_atoms, protein_residues = protein_complex["ligand"], protein_complex["protein_atoms"], protein_complex["protein_residues"]
        sample = mol_to_feature(ligand_mol=ligand, target_mol=protein_atoms)

        ligand = Chem.RemoveHs(ligand)
        ligand_graph = mol_to_bigraph(mol=ligand,
                                      node_featurizer=self.node_featurizer, 
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=self.add_self_loop)

        protein_residue_coords = []
        residue_smiles = []  # SMILES strings of residues in the protein
        atom_to_residue = {}
        residue_counter = 0
        for res in protein_residues.get_residues():
            # if is_aa(res):
            protein_residue_coords.append(utils.get_atom_coords(res))
            res_mol = utils.residue_to_mol(res)
            residue_smiles.append(Chem.MolToSmiles(res_mol))
            for atom in res:
                atom_to_residue[tuple([round(x, 2) for x in atom.get_coord().tolist()])] = (residue_counter, atom.get_id(), res.get_resname())
            residue_counter += 1


        # backbone ["N", "CA", "C", "O"] coordinates for proteins
        # shape: [seq_len, 4, 3]
        protein_residue_coords = torch.as_tensor(
            np.asarray(protein_residue_coords), dtype=torch.float32
        )

        # SMILES strings of AA residues and ligand
        # in the same order with the nodes in the graph
        smiles_strings = residue_smiles
        if self.residue_featurizer:
            residues = (
                torch.stack(
                    [
                        self.residue_featurizer.featurize(smiles)
                        for smiles in smiles_strings
                    ]
                )
                .to(self.device)
                .to(torch.long)
            )
            # shape: [seq_len + 1, d_embed]
        
        # construct knn graph from C-alpha coordinates
        ca_coords = protein_residue_coords[:, 1]
        protein_graph = dgl.knn_graph(ca_coords, k=min(self.top_k, ca_coords.shape[0]))
        ca_edge_index = protein_graph.edges()

        pos_embeddings = self._positional_embeddings(ca_edge_index)
        ca_vectors = ca_coords[ca_edge_index[0]] - ca_coords[ca_edge_index[1]]
        rbf = _rbf(
            ca_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        dihedrals = self._dihedrals(protein_residue_coords)
        orientations = self._orientations(ca_coords)
        sidechains = self._sidechains(protein_residue_coords)

        if self.residue_featurizer:
            node_s = torch.cat([dihedrals, residues], dim=-1)
        else:
            node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(ca_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        protein_graph.ndata["node_s"] = node_s
        protein_graph.ndata["node_v"] = node_v
        # edge features
        protein_graph.edata["edge_s"] = edge_s
        protein_graph.edata["edge_v"] = edge_v

        # shape: [ligand_n_atoms, 3]
        ligand_coords = torch.as_tensor(
            ligand.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        ligand_vectors = ligand_coords[ligand_graph.edges()[0].long()] - ligand_coords[ligand_graph.edges()[1].long()]
        # ligand node features
        ligand_graph.ndata["node_s"] = ligand_graph.ndata["h"]
        ligand_graph.ndata["node_v"] = ligand_coords.unsqueeze(-2)
        # ligand edge features
        ligand_graph.edata["edge_s"] = ligand_graph.edata["e"]
        ligand_graph.edata["edge_v"] = _normalize(ligand_vectors).unsqueeze(-2)

        # construct complex graph from rdkit computed interactions
        n_ligand = sample['interaction_indice'].shape[1]
        n_protein = sample['interaction_indice'].shape[2]
        interaction_indice_pad = np.pad(sample['interaction_indice'], 
                                [(0, 0), (n_protein, 0), (0, n_ligand)])
        interaction_indice_symm = np.maximum(interaction_indice_pad, np.transpose(interaction_indice_pad, (0, 2, 1)))
        src, dst = np.nonzero(interaction_indice_symm.max(axis=0) + np.eye(n_protein+n_ligand))
        complex_graph = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)))
        edge_index = complex_graph.edges()

        # combine protein and ligand atomic coordinates
        
        # shape: [protein_n_atoms, 3]
        protein_atom_coords = torch.as_tensor(
            protein_atoms.GetConformers()[0].GetPositions(), dtype=torch.float32
        )

        X_ca = torch.cat((protein_atom_coords, ligand_coords), axis=0)

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(
            E_vectors.norm(dim=-1),
            D_count=self.num_rbf,
            device=self.device,
        )

        node_s = 1.*(torch.cat([torch.from_numpy(sample['target_h']), 
                            torch.from_numpy(sample['ligand_h'])], dim=0)).float()
        node_v = torch.cat([torch.from_numpy(sample['target_pos']), 
                            torch.from_numpy(sample['ligand_pos'])], dim=0).unsqueeze(-2).float()
        edge_s = torch.from_numpy(interaction_indice_symm[:, src, dst]).T.float()
        edge_v = _normalize(E_vectors).unsqueeze(-2).float()

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
        )

        # node features
        complex_graph.ndata["node_s"] = node_s
        complex_graph.ndata["node_v"] = node_v
        # edge features
        complex_graph.edata["edge_s"] = edge_s
        complex_graph.edata["edge_v"] = edge_v

        if self.residue_featurizer:
            return protein_graph, ligand_graph, complex_graph, sample, atom_to_residue
        else:
            return protein_graph, ligand_graph, complex_graph, sample, atom_to_residue, smiles_strings