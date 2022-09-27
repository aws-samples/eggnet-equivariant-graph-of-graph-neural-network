from typing import Any, Callable, Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gvp import GVP, GVPConvLayer, LayerNorm

import dgl

# from transformers import BertModel
from dgl.nn import GATConv


ATOMIC_KEYS = [
    "C",
    "CA",
    "CB",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CG",
    "CG1",
    "CG2",
    "CH2",
    "CZ",
    "CZ2",
    "CZ3",
    "N",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NI",
    "NZ",
    "O",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "SD",
    "SG",
    "MG",
    "MN",
    "ZN",
    "FE",
    "NA",
    "K",
]


def padded_stack(
    tensors: List[torch.Tensor],
    side: str = "right",
    mode: str = "constant",
    value: Union[int, float] = 0,
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(
                x, make_padding(full_size - x.size(-1)), mode=mode, value=value
            )
            if full_size - x.size(-1) > 0
            else x
            for x in tensors
        ],
        dim=0,
    )
    return out


class GVPEncoder(nn.Module):
    """
    GVP-GNN model without the final dense layer.
    The encoder will return a tensor with shape: [n_nodes, node_h_dim[0]]
    """

    def __init__(
        self,
        node_in_dim: Tuple[int, int],
        node_h_dim: Tuple[int, int],
        edge_in_dim: Tuple[int, int],
        edge_h_dim: Tuple[int, int],
        num_layers=3,
        drop_rate=0.1,
        residual=True,
        seq_embedding=True,
    ):
        """Initializes the module
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            seq_embedding: whether to one-hot embed the sequence
        Returns:
            None
        """
        super(GVPEncoder, self).__init__()
        self.residual = residual
        self.seq_embedding = seq_embedding

        if seq_embedding:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            node_h_dim = (
                node_h_dim[0] * num_layers,
                node_h_dim[1] * num_layers,
            )
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0))
        )

    def forward(self, g):
        """Helper function to perform GVP network forward pass.
        Args:
            g: dgl.graph
        Returns:
            node embeddings
        """
        h_V = (g.ndata["node_s"], g.ndata["node_v"])
        h_E = (g.edata["edge_s"], g.edata["edge_v"])
        if self.seq_embedding:
            seq = g.ndata["seq"]
            # one-hot encodings
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        g.ndata["node_s"], g.ndata["node_v"] = h_V[0], h_V[1]
        g.edata["edge_s"], g.edata["edge_v"] = h_E[0], h_E[1]
        # GVP Conv layers
        if not self.residual:
            for layer in self.layers:
                h_V = layer(g)
            out = self.W_out(h_V)
        else:
            h_V_out = []  # collect outputs from all GVP Conv layers
            for layer in self.layers:
                h_V = layer(g)
                h_V_out.append(h_V)
                g.ndata["node_s"], g.ndata["node_v"] = h_V[0], h_V[1]
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out = (
                torch.cat([h_V[0] for h_V in h_V_out], dim=-1),
                torch.cat([h_V[1] for h_V in h_V_out], dim=-2),
            )
            out = self.W_out(h_V_out)
        return out  # shape: [n_nodes, ns]


class GVPModel(nn.Module):
    """GVP-GNN Model (structure-only) modified from `MQAModel`:
    https://github.com/drorlab/gvp-pytorch/blob/main/gvp/model.py
    """

    def __init__(
        self,
        node_in_dim: Tuple[int, int],
        node_h_dim: Tuple[int, int],
        edge_in_dim: Tuple[int, int],
        edge_h_dim: Tuple[int, int],
        num_layers=3,
        drop_rate=0.1,
        residual=True,
        num_outputs=1,
        seq_embedding=True,
        is_hetero=False,
        use_energy_decoder=False,
        vdw_N=6.0,
        max_vdw_interaction=0.0356,
        min_vdw_interaction=0.0178,
        dev_vdw_radius=0.2,
        no_rotor_penalty=False,
        **kwargs,
    ):
        """Initializes the module
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            num_outputs: number of output units
            seq_embedding: whether to one-hot embed the sequence
        Returns:
            None
        """
        super(GVPModel, self).__init__()
        self.gvp_encoder = GVPEncoder(
            node_in_dim,
            node_h_dim,
            edge_in_dim,
            edge_h_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
            residual=residual,
            seq_embedding=seq_embedding,
        )
        self.num_outputs = num_outputs
        self.use_energy_decoder = use_energy_decoder
        self.is_hetero = is_hetero
        ns, nv = node_h_dim
        ## Decoder
        if use_energy_decoder:
            self.decoder = EnergyDecoder(
                ns,
                vdw_N=vdw_N,
                max_vdw_interaction=max_vdw_interaction,
                min_vdw_interaction=min_vdw_interaction,
                dev_vdw_radius=dev_vdw_radius,
                no_rotor_penalty=no_rotor_penalty,
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(ns, 2 * ns),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
                nn.Linear(2 * ns, self.num_outputs),
            )

    def forward(
        self,
        g,
        sample=None,
        DM_min=0.5,
        cal_der_loss=False,
    ):
        """Perform GVP network forward pass.
        Args:
            g: dgl.graph
            sample: dict, output from `mol_to_feature`
        Returns:
            (logits, g_logits)
        """
        out = self.gvp_encoder(g)  # shape: [n_nodes, ns]
        # aggregate node vectors to graph
        g.ndata["out"] = out
        ## Decoder
        if self.use_energy_decoder:
            # split and reshape node embeddings to separate those from
            # protein and ligand
            out_protein = []
            out_ligand = []
            for graph in dgl.unbatch(g):
                ligand_mask = graph.ndata["mask"] == 0
                ligand_out_h = graph.ndata["out"][ligand_mask].permute(1, 0)
                protein_mask = graph.ndata["mask"] == 1
                protein_out_h = graph.ndata["out"][protein_mask].permute(1, 0)
                # repeat by number of atoms
                ligand_out_h_atoms = ligand_out_h.repeat(
                    1, graph.ndata["atom_counts"][ligand_mask]
                )
                protein_n_atoms = graph.ndata["atom_counts"][protein_mask]
                protein_out_h_atoms = []
                for i in torch.arange(len(protein_n_atoms)):
                    protein_out_h_i = protein_out_h[:, [i]]
                    protein_out_h_atoms.append(
                        protein_out_h_i.repeat(1, protein_n_atoms[i])
                    )
                protein_out_h_atoms = torch.cat(protein_out_h_atoms, axis=1)
                out_ligand.append(ligand_out_h_atoms)
                out_protein.append(protein_out_h_atoms)

            target_h = padded_stack(out_protein).permute(
                0, 2, 1
            )  # dim: [batch_size, max_atoms_protein, ns]
            ligand_h = padded_stack(out_ligand).permute(
                0, 2, 1
            )  # dim: [batch_size, max_atoms_ligand, ns]

            # concat features
            h1_ = ligand_h.unsqueeze(2).repeat(
                1, 1, target_h.size(1), 1
            )  # dim: [batch_size, max_atoms_ligand, max_atoms_protein, ns]
            h2_ = target_h.unsqueeze(1).repeat(
                1, ligand_h.size(1), 1, 1
            )  # dim: [batch_size, max_atoms_ligand, max_atoms_protein, ns]
            h_cat = torch.cat(
                [h1_, h2_], -1
            )  # dim: [batch_size, max_atoms_ligand, max_atoms_protein, ns*2]

            return self.decoder(
                sample, h_cat, DM_min=DM_min, cal_der_loss=cal_der_loss
            )
        else:
            # aggregate node vectors to graph
            graph_out = dgl.mean_nodes(g, "out")  # [n_graphs, ns]
            return self.decoder(out) + 0.5, self.decoder(graph_out) + 0.5


class GATModel(nn.Module):
    """GAT structure-only model."""

    def __init__(
        self,
        in_feats=6,
        h_dim=128,
        num_layers=3,
        n_hidden=512,
        drop_rate=0.2,
        num_outputs=1,
        seq_embedding=True,
        **kwargs,
    ):
        """Initializes the model
        Args:
            in_feats: dim of the node scalar features
            h_dim: dim of the output layer for GATConv
            num_heads: number of attention heads for GATConv
            num_layers: number of GATConv layers
            n_hidden: number of hidden units in classification head
            drop_rate: rate to use in the dropout layer
            num_outputs: number of output units
            seq_embedding: whether to one-hot embed the sequence
        Returns:
            None
        """
        super(GATModel, self).__init__()

        self.seq_embedding = seq_embedding
        self.embeding_dim = 20
        # input dim for the 2nd layer forward
        next_in_feats = kwargs["num_heads"] * h_dim
        self.next_in_feats = next_in_feats

        if seq_embedding:
            # one-hot encoding for AAs
            self.W_s = nn.Embedding(self.embeding_dim, self.embeding_dim)
            node_in_feats = in_feats + self.embeding_dim
        else:
            node_in_feats = in_feats

        # GAT layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = GATConv(node_in_feats, h_dim, **kwargs)
            else:
                layer = GATConv(next_in_feats, h_dim, **kwargs)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate)

        self.dense = nn.Sequential(
            nn.Linear(next_in_feats, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, num_outputs),
        )

    def forward(self, batch):
        """Perform the forward pass.
        Args:
            batch: dgl.graph
        Returns:
            (node_logits, graph_logits)
        """
        logits, g_logits = self._forward(batch)
        return logits, g_logits

    def _forward(self, g):
        """Helper function to perform the forward pass.
        Args:
            g: dgl.graph
        Returns:
            (node_logits, graph_logits)
        """
        seq = g.ndata["seq"]
        # one-hot encodings
        node_embeddings = self.W_s(seq)  # [n_nodes, 20]
        # GAT forward
        for i, layer in enumerate(self.layers):
            if i == 0:
                gat_out = layer(
                    g, torch.cat((node_embeddings, g.ndata["node_s"]), axis=1)
                )
            else:
                gat_out = layer(g, gat_out)
            gat_out = gat_out.view(-1, self.next_in_feats)

        out = self.dropout(self.relu(gat_out))  # [n_nodes, next_in_feats]
        out = self.dense(out) + 0.5  # [n_nodes, num_outputs]

        # aggregate node vectors to graph
        g.ndata["out"] = out
        graph_out = dgl.mean_nodes(g, "out")  # [bs, next_in_feats]

        return out, graph_out


class MultiStageGVPModel(nn.Module):
    """GVP-GNN Model (structure-only) modified from `MQAModel`:
    https://github.com/drorlab/gvp-pytorch/blob/main/gvp/model.py
    """

    def __init__(
        self,
        protein_node_in_dim: Tuple[int, int],
        protein_edge_in_dim: Tuple[int, int],
        ligand_node_in_dim: Tuple[int, int],
        ligand_edge_in_dim: Tuple[int, int],
        complex_edge_in_dim: Tuple[int, int],
        stage1_node_h_dim: Tuple[int, int],
        stage1_edge_h_dim: Tuple[int, int],
        stage2_node_h_dim: Tuple[int, int],
        stage2_edge_h_dim: Tuple[int, int],
        stage1_num_layers=3,
        stage2_num_layers=3,
        drop_rate=0.1,
        residual=True,
        num_outputs=1,
        seq_embedding=True,
        is_hetero=False,
        use_energy_decoder=False,
        vdw_N=6.0,
        max_vdw_interaction=0.0356,
        min_vdw_interaction=0.0178,
        dev_vdw_radius=0.2,
        no_rotor_penalty=False,
        **kwargs,
    ):
        """Initializes the module
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            num_outputs: number of output units
            seq_embedding: whether to one-hot embed the sequence
        Returns:
            None
        """
        super(MultiStageGVPModel, self).__init__()
        self.residual = residual
        self.num_outputs = num_outputs
        self.seq_embedding = seq_embedding
        self.use_energy_decoder = use_energy_decoder
        self.is_hetero = is_hetero

        if seq_embedding:
            self.W_s_p = nn.Embedding(20, 20)
            protein_node_in_dim = (
                protein_node_in_dim[0] + 20,
                protein_node_in_dim[1],
            )

        protein_node_h_dim = stage1_node_h_dim
        ligand_node_h_dim = stage1_node_h_dim
        protein_edge_h_dim = stage1_edge_h_dim
        ligand_edge_h_dim = stage1_edge_h_dim
        protein_num_layers = stage1_num_layers
        ligand_num_layers = stage1_num_layers
        complex_node_in_dim = stage1_node_h_dim
        complex_node_h_dim = stage2_node_h_dim
        complex_edge_h_dim = stage2_edge_h_dim
        complex_num_layers = stage2_num_layers

        ## FIRST STAGE

        ## Protein branch
        self.W_v_p = nn.Sequential(
            LayerNorm(protein_node_in_dim),
            GVP(
                protein_node_in_dim,
                protein_node_h_dim,
                activations=(None, None),
            ),
        )
        self.W_e_p = nn.Sequential(
            LayerNorm(protein_edge_in_dim),
            GVP(
                protein_edge_in_dim,
                protein_edge_h_dim,
                activations=(None, None),
            ),
        )

        self.protein_layers = nn.ModuleList(
            GVPConvLayer(
                protein_node_h_dim, protein_edge_h_dim, drop_rate=drop_rate
            )
            for _ in range(protein_num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            protein_node_h_dim = (
                protein_node_h_dim[0] * protein_num_layers,
                protein_node_h_dim[1] * protein_num_layers,
            )
        ns_p, _ = protein_node_h_dim
        self.W_out_p = nn.Sequential(
            LayerNorm(protein_node_h_dim), GVP(protein_node_h_dim, (ns_p, 0))
        )

        ## Ligand branch
        self.W_v_l = nn.Sequential(
            LayerNorm(ligand_node_in_dim),
            GVP(
                ligand_node_in_dim, ligand_node_h_dim, activations=(None, None)
            ),
        )
        self.W_e_l = nn.Sequential(
            LayerNorm(ligand_edge_in_dim),
            GVP(
                ligand_edge_in_dim, ligand_edge_h_dim, activations=(None, None)
            ),
        )

        self.ligand_layers = nn.ModuleList(
            GVPConvLayer(
                ligand_node_h_dim, ligand_edge_h_dim, drop_rate=drop_rate
            )
            for _ in range(ligand_num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            ligand_node_h_dim = (
                ligand_node_h_dim[0] * ligand_num_layers,
                ligand_node_h_dim[1] * ligand_num_layers,
            )
        ns_l, _ = ligand_node_h_dim
        self.W_out_l = nn.Sequential(
            LayerNorm(ligand_node_h_dim), GVP(ligand_node_h_dim, (ns_l, 0))
        )

        ## SECOND STAGE

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            complex_node_in_dim = (
                complex_node_in_dim[0] * stage1_num_layers,
                complex_node_in_dim[1] * stage1_num_layers,
            )

        ## Complex branch
        self.W_v_c = nn.Sequential(
            LayerNorm(complex_node_in_dim),
            GVP(
                complex_node_in_dim,
                complex_node_h_dim,
                activations=(None, None),
            ),
        )
        self.W_e_c = nn.Sequential(
            LayerNorm(complex_edge_in_dim),
            GVP(
                complex_edge_in_dim,
                complex_edge_h_dim,
                activations=(None, None),
            ),
        )

        self.complex_layers = nn.ModuleList(
            GVPConvLayer(
                complex_node_h_dim, complex_edge_h_dim, drop_rate=drop_rate
            )
            for _ in range(complex_num_layers)
        )

        if self.residual:
            # concat outputs from GVPConvLayer(s)
            complex_node_h_dim = (
                complex_node_h_dim[0] * complex_num_layers,
                complex_node_h_dim[1] * complex_num_layers,
            )
        ns_c, nv_c = complex_node_h_dim
        self.W_out_c = nn.Sequential(
            LayerNorm(complex_node_h_dim), GVP(complex_node_h_dim, (ns_c, 0))
        )

        ## Decoder
        if use_energy_decoder:
            if self.is_hetero:
                self.atomic_decomposition_s = nn.ModuleDict(
                    {
                        atomic_key: nn.Sequential(
                            nn.Linear(ns_c, 2 * ns_c),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=drop_rate),
                            nn.Linear(2 * ns_c, ns_c),
                        )
                        for atomic_key in ATOMIC_KEYS + ["Other"]
                    }
                )

                self.atomic_decomposition_v = nn.ModuleDict(
                    {
                        atomic_key: nn.Sequential(
                            nn.Linear(nv_c, 2 * nv_c),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=drop_rate),
                            nn.Linear(2 * nv_c, nv_c),
                        )
                        for atomic_key in ATOMIC_KEYS + ["Other"]
                    }
                )

            self.decoder = EnergyDecoder(
                ns_c,
                vdw_N=vdw_N,
                max_vdw_interaction=max_vdw_interaction,
                min_vdw_interaction=min_vdw_interaction,
                dev_vdw_radius=dev_vdw_radius,
                no_rotor_penalty=no_rotor_penalty,
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(ns_c, 2 * ns_c),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
                nn.Linear(2 * ns_c, self.num_outputs),
            )

    def forward(
        self,
        protein_batch,
        ligand_batch,
        complex_batch,
        sample=None,
        DM_min=0.5,
        cal_der_loss=False,
        atom_to_residue=None,
    ):
        """Perform the forward pass.
        Args:
            batch: dgl.DGLGraph
        Returns:
            (logits, g_logits)
        """
        if self.use_energy_decoder:
            energies, der1, der2 = self._forward(
                protein_batch,
                ligand_batch,
                complex_batch,
                sample,
                DM_min,
                cal_der_loss,
                atom_to_residue,
            )
            return energies, der1, der2
        else:
            logits, g_logits = self._forward(
                protein_batch, ligand_batch, complex_batch
            )
            return logits, g_logits

    def _forward(
        self,
        protein_graph,
        ligand_graph,
        complex_graph,
        sample=None,
        DM_min=0.5,
        cal_der_loss=False,
        atom_to_residue=None,
    ):
        """Helper function to perform GVP network forward pass.
        Args:
            g: dgl.graph
        Returns:
            (logits, g_logits)
        """

        ## FIRST STAGE

        ## Protein branch
        h_V_p = (protein_graph.ndata["node_s"], protein_graph.ndata["node_v"])
        h_E_p = (protein_graph.edata["edge_s"], protein_graph.edata["edge_v"])
        if self.seq_embedding:
            seq_p = protein_graph.ndata["seq"]
            # one-hot encodings
            seq_p = self.W_s_p(seq_p)
            h_V_p = (torch.cat([h_V_p[0], seq_p], dim=-1), h_V_p[1])

        h_V_p = self.W_v_p(h_V_p)
        h_E_p = self.W_e_p(h_E_p)
        protein_graph.ndata["node_s"], protein_graph.ndata["node_v"] = (
            h_V_p[0],
            h_V_p[1],
        )
        protein_graph.edata["edge_s"], protein_graph.edata["edge_v"] = (
            h_E_p[0],
            h_E_p[1],
        )
        # GVP Conv layers
        if not self.residual:
            for protein_layer in self.protein_layers:
                h_V_p = protein_layer(protein_graph)
        else:
            h_V_out_p = []  # collect outputs from all GVP Conv layers
            for protein_layer in self.protein_layers:
                h_V_p = protein_layer(protein_graph)
                h_V_out_p.append(h_V_p)
                (
                    protein_graph.ndata["node_s"],
                    protein_graph.ndata["node_v"],
                ) = (h_V_p[0], h_V_p[1])
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out_p = (
                torch.cat([h_V_p[0] for h_V_p in h_V_out_p], dim=-1),
                torch.cat([h_V_p[1] for h_V_p in h_V_out_p], dim=-2),
            )

        ## Ligand branch
        h_V_l = (ligand_graph.ndata["node_s"], ligand_graph.ndata["node_v"])
        h_E_l = (ligand_graph.edata["edge_s"], ligand_graph.edata["edge_v"])

        h_V_l = self.W_v_l(h_V_l)
        h_E_l = self.W_e_l(h_E_l)
        ligand_graph.ndata["node_s"], ligand_graph.ndata["node_v"] = (
            h_V_l[0],
            h_V_l[1],
        )
        ligand_graph.edata["edge_s"], ligand_graph.edata["edge_v"] = (
            h_E_l[0],
            h_E_l[1],
        )
        # GVP Conv layers
        if not self.residual:
            for ligand_layer in self.ligand_layers:
                h_V_l = ligand_layer(ligand_graph)
        else:
            h_V_out_l = []  # collect outputs from all GVP Conv layers
            for ligand_layer in self.ligand_layers:
                h_V_l = ligand_layer(ligand_graph)
                h_V_out_l.append(h_V_l)
                ligand_graph.ndata["node_s"], ligand_graph.ndata["node_v"] = (
                    h_V_l[0],
                    h_V_l[1],
                )
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out_l = (
                torch.cat([h_V_l[0] for h_V_l in h_V_out_l], dim=-1),
                torch.cat([h_V_l[1] for h_V_l in h_V_out_l], dim=-2),
            )

        ## SECOND STAGE

        protein_num_nodes = protein_graph.batch_num_nodes().tolist()
        ligand_num_nodes = ligand_graph.batch_num_nodes().tolist()
        complex_num_nodes = complex_graph.batch_num_nodes().tolist()

        h_V_p_s = torch.split(
            h_V_out_p[0] if self.residual else h_V_p[0], protein_num_nodes
        )
        h_V_p_v = torch.split(
            h_V_out_p[1] if self.residual else h_V_p[1], protein_num_nodes
        )
        h_V_l_s = torch.split(
            h_V_out_l[0] if self.residual else h_V_l[0], ligand_num_nodes
        )
        h_V_l_v = torch.split(
            h_V_out_l[1] if self.residual else h_V_l[1], ligand_num_nodes
        )

        if self.is_hetero and self.use_energy_decoder:
            protein_num_nodes_temp = []
            complex_v = torch.split(
                complex_graph.ndata["node_v"], complex_num_nodes
            )
            h_V_p_s_temp, h_V_p_v_temp = [], []
            for i, (cv, nl) in enumerate(zip(complex_v, ligand_num_nodes)):
                protein_s = h_V_p_s[i]
                protein_v = h_V_p_v[i]
                residue_lookup = atom_to_residue[i]
                protein_atom_coords = cv.squeeze(1)[:-nl]
                protein_atom_s_list, protein_atom_v_list = [], []
                num_atoms = 0
                for coords in protein_atom_coords:
                    k = tuple([round(j, 2) for j in coords.tolist()])
                    residue_idx, atom_id, res_name = residue_lookup[k]
                    # atom_type = atom_id[0]
                    if atom_id in ATOMIC_KEYS:
                        protein_atom_s = self.atomic_decomposition_s[atom_id](
                            protein_s[residue_idx, :]
                        )
                        protein_atom_v = self.atomic_decomposition_v[atom_id](
                            protein_v[residue_idx, :].permute(1, 0)
                        )
                    else:
                        protein_atom_s = self.atomic_decomposition_s["Other"](
                            protein_s[residue_idx, :]
                        )
                        protein_atom_v = self.atomic_decomposition_v["Other"](
                            protein_v[residue_idx, :].permute(1, 0)
                        )
                    # protein_atom_v = protein_v[residue_idx, :]
                    protein_atom_s_list.append(protein_atom_s)
                    protein_atom_v_list.append(protein_atom_v.permute(1, 0))
                    num_atoms += 1
                h_V_p_s_temp.append(torch.stack(protein_atom_s_list))
                h_V_p_v_temp.append(torch.stack(protein_atom_v_list))
                protein_num_nodes_temp.append(num_atoms)
            h_V_p_s = h_V_p_s_temp
            h_V_p_v = h_V_p_v_temp
            protein_num_nodes = protein_num_nodes_temp

        h_V_s = [val for pair in zip(h_V_p_s, h_V_l_s) for val in pair]
        h_V_v = [val for pair in zip(h_V_p_v, h_V_l_v) for val in pair]

        complex_graph.ndata["node_s"] = torch.cat(h_V_s, dim=0)
        complex_graph.ndata["node_v"] = torch.cat(h_V_v, dim=0)

        ## Complex branch
        h_V_c = (complex_graph.ndata["node_s"], complex_graph.ndata["node_v"])
        h_E_c = (complex_graph.edata["edge_s"], complex_graph.edata["edge_v"])

        h_V_c = self.W_v_c(h_V_c)
        h_E_c = self.W_e_c(h_E_c)
        complex_graph.ndata["node_s"], complex_graph.ndata["node_v"] = (
            h_V_c[0],
            h_V_c[1],
        )
        complex_graph.edata["edge_s"], complex_graph.edata["edge_v"] = (
            h_E_c[0],
            h_E_c[1],
        )
        # GVP Conv layers
        if not self.residual:
            for complex_layer in self.complex_layers:
                h_V_c = complex_layer(complex_graph)
            out_c = self.W_out_c(h_V_c)
        else:
            h_V_out_c = []  # collect outputs from all GVP Conv layers
            for complex_layer in self.complex_layers:
                h_V_c = complex_layer(complex_graph)
                h_V_out_c.append(h_V_c)
                (
                    complex_graph.ndata["node_s"],
                    complex_graph.ndata["node_v"],
                ) = (h_V_c[0], h_V_c[1])
            # concat outputs from GVPConvLayers (separatedly for s and V)
            h_V_out_c = (
                torch.cat([h_V_c[0] for h_V_c in h_V_out_c], dim=-1),
                torch.cat([h_V_c[1] for h_V_c in h_V_out_c], dim=-2),
            )
            out_c = self.W_out_c(h_V_out_c)

        ## Decoder
        if self.use_energy_decoder:
            protein_ligand_num_nodes = [
                val
                for pair in zip(protein_num_nodes, ligand_num_nodes)
                for val in pair
            ]
            assert sum(complex_num_nodes) == sum(protein_ligand_num_nodes)

            out_c_split = torch.split(out_c, protein_ligand_num_nodes)
            out_c_protein = [x.permute(1, 0) for x in out_c_split[::2]]
            out_c_ligand = [x.permute(1, 0) for x in out_c_split[1::2]]

            target_h = padded_stack(out_c_protein).permute(
                0, 2, 1
            )  # dim: [batch_size, max_atoms_protein, ns_protein]
            ligand_h = padded_stack(out_c_ligand).permute(
                0, 2, 1
            )  # dim: [batch_size, max_atoms_ligand, ns_ligand]

            # concat features
            h1_ = ligand_h.unsqueeze(2).repeat(
                1, 1, target_h.size(1), 1
            )  # dim: [batch_size, max_atoms_ligand, max_atoms_protein, ns_ligand]
            h2_ = target_h.unsqueeze(1).repeat(
                1, ligand_h.size(1), 1, 1
            )  # dim: [batch_size, max_atoms_ligand, max_atoms_protein, ns_protein]
            h_cat = torch.cat(
                [h1_, h2_], -1
            )  # dim: [batch_size, max_atoms_ligand, max_atoms_protein, ns_ligand+ns_protein]

            return self.decoder(
                sample, h_cat, DM_min=DM_min, cal_der_loss=cal_der_loss
            )
        else:
            # aggregate node vectors to graph
            complex_graph.ndata["out"] = out_c
            graph_out_c = dgl.mean_nodes(
                complex_graph, "out"
            )  # [n_graphs, ns]

            return self.decoder(out_c) + 0.5, self.decoder(graph_out_c) + 0.5


class EnergyDecoder(nn.Module):
    """
    Reference: https://github.com/ACE-KAIST/PIGNet/blob/main/models.py
    """

    def __init__(
        self,
        ns_c,
        vdw_N=6.0,
        max_vdw_interaction=0.0356,
        min_vdw_interaction=0.0178,
        dev_vdw_radius=0.2,
        no_rotor_penalty=False,
        **kwargs,
    ):
        """Initializes the module
        Args:
            node_in_dim: node dimensions (s, V) in input graph
            node_h_dim: node dimensions to use in GVP-GNN layers
            edge_in_dim: edge dimensions (s, V) in input graph
            edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers
            num_layers: number of GVP-GNN layers
            drop_rate: rate to use in all dropout layers
            residual: whether to have residual connections among GNN layers
            num_outputs: number of output units
            seq_embedding: whether to one-hot embed the sequence
        Returns:
            None
        """
        super(EnergyDecoder, self).__init__()
        self.vdw_N = vdw_N
        self.max_vdw_interaction = max_vdw_interaction
        self.min_vdw_interaction = min_vdw_interaction
        self.dev_vdw_radius = dev_vdw_radius
        self.no_rotor_penalty = no_rotor_penalty

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(ns_c * 2, ns_c),
            nn.ReLU(),
            nn.Linear(ns_c, 1),
            nn.Sigmoid(),
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(ns_c * 2, ns_c),
            nn.ReLU(),
            nn.Linear(ns_c, 1),
            nn.Tanh(),
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(ns_c * 2, ns_c),
            nn.ReLU(),
            nn.Linear(ns_c, 1),
            nn.Sigmoid(),
        )
        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))

    def cal_hbond(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hbond_coeff * self.hbond_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_hydrophobic(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_vdw_interaction(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        ligand_valid: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(
            1, 1, target_valid.size(1)
        )
        target_valid_ = target_valid.unsqueeze(1).repeat(
            1, ligand_valid.size(1), 1
        )
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )

        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm_0[dm_0 < 0.0001] = 1
        N = self.vdw_N
        vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.max_vdw_interaction - self.min_vdw_interaction)
        A = A + self.min_vdw_interaction

        energy = vdw_term1 + vdw_term2
        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_
        energy = A * energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float
    ) -> Tensor:
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(self, sample, h_cat, DM_min=0.5, cal_der_loss=False):
        """Perform the forward pass.
        Args:
            sample: Dict of tensors created by pignet_featurizers
            h_cat: torch.Tensor batch of hidden representations of atoms
        Returns:
            energies, der1, der2
        """
        (
            ligand_h,
            ligand_adj,
            target_h,
            target_adj,
            interaction_indice,
            ligand_pos,
            target_pos,
            rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_valid,
            target_valid,
            ligand_non_metal,
            target_non_metal,
        ) = sample.values()

        # distance matrix
        ligand_pos.requires_grad = True
        dm = self.cal_distance_matrix(
            ligand_pos, target_pos, DM_min
        )  # [batch_size, n_nodes_ligand, n_nodes_target]

        # compute energy component
        energies = []

        # vdw interaction
        vdw_energy = self.cal_vdw_interaction(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
        )
        energies.append(vdw_energy)

        # hbond interaction
        hbond = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 0],
        )
        energies.append(hbond)

        # metal interaction
        metal = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 1],
        )
        energies.append(metal)

        # hydrophobic interaction
        hydrophobic = self.cal_hydrophobic(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 2],
        )
        energies.append(hydrophobic)

        energies = torch.cat(energies, -1)
        # rotor penalty
        if not self.no_rotor_penalty:
            energies = energies / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        # derivatives
        if cal_der_loss:
            gradient = torch.autograd.grad(
                energies.sum(),
                ligand_pos,
                retain_graph=True,
                create_graph=True,
            )[0]
            der1 = torch.pow(gradient.sum(1), 2).mean()
            der2 = torch.autograd.grad(
                gradient.sum(),
                ligand_pos,
                retain_graph=True,
                create_graph=True,
            )[0]
            der2 = -der2.sum(1).sum(1).mean()
        else:
            der1 = torch.zeros_like(energies).sum()
            der2 = torch.zeros_like(energies).sum()

        return energies, der1, der2


class EigenDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super(EigenDecoder, self).__init__()
        self.energy_coeff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        # X dim: [batch_size, num_atoms, hidden_dim]
        E_batch = []
        for i in range(X.shape[0]):
            Xi = X[i, ...]
            Ci = self.energy_coeff(Xi).squeeze(1)
            Ai = torch.matmul(Xi, Xi.T)  # dim: [num_atoms, num_atoms]
            Li, Vi = torch.linalg.eig(Ai)  # Eigendecomposition
            # Li dim: [num_atoms, num_atoms] (diagonal of eigenvalues)
            # Vi dim: [num_atoms, num_atoms] (stacked eigenvectors)
            Zi = torch.matmul(Vi.T, Li)  # dim: [num_atoms]
            Ai_complex = torch.complex(Ai, torch.zeros_like(Ai))
            deltaZi = Zi - torch.matmul(Vi.T, torch.mean(Ai_complex, axis=1))
            Ei = Ci * torch.real(
                torch.square(deltaZi) / Li
            )  # dim: [num_atoms]
            E_batch.append(Ei)
        return torch.stack(E_batch)  # dim: [batch_size, num_atoms]
