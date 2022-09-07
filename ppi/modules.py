from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gvp import GVP, GVPConvLayer, LayerNorm

import dgl

# from transformers import BertModel
from dgl.nn import GATConv


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
        self.residual = residual
        self.num_outputs = num_outputs
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

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, self.num_outputs),
        )

    def forward(self, batch):
        """Perform the forward pass.
        Args:
            batch: dgl.DGLGraph
        Returns:
            (logits, g_logits)
        """
        logits, g_logits = self._forward(batch)
        return logits, g_logits

    def _forward(self, g):
        """Helper function to perform GVP network forward pass.
        Args:
            g: dgl.graph
        Returns:
            (logits, g_logits)
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
        # out.shape # [n_nodes, ns]
        # aggregate node vectors to graph
        g.ndata["out"] = out
        graph_out = dgl.mean_nodes(g, "out")  # [n_graphs, ns]

        return self.dense(out) + 0.5, self.dense(graph_out) + 0.5


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
