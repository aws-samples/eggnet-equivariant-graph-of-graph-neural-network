import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import GVPModel, GVPMultiStageModel


class LitGVPModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        hparams = [
            "lr",
            "node_in_dim",
            "node_h_dim",
            "edge_in_dim",
            "edge_h_dim",
            "num_layers",
            "drop_rate",
            "residual",
            "seq_embedding",
        ]
        self.save_hyperparameters(*hparams)
        model_kwargs = {key: kwargs[key] for key in hparams if key in kwargs}
        self.model = GVPModel(**model_kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model specific args to the base/parent parser.
        Args:
            parent_parser: Base/parent parser
        Returns:
            parent parser with additional model-specific args
        """
        parser = parent_parser.add_argument_group("GVPModel")
        parser.add_argument(
            "--node_h_dim",
            type=int,
            nargs="+",
            default=(100, 16),
            help="node_h_dim in GVP",
        )
        parser.add_argument(
            "--edge_h_dim",
            type=int,
            nargs="+",
            default=(32, 1),
            help="edge_h_dim in GVP",
        )

        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--drop_rate", type=float, default=0.1)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--seq_embedding", action="store_true")
        parser.set_defaults(residual=False, seq_embedding=False)
        return parent_parser

    def _compute_loss(self, logits, targets):
        # binary classification
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        # regression
        loss = F.mse_loss(logits, targets)
        return loss

    def forward(self, g):
        return self.model(g)

    def _step(self, batch, batch_idx, prefix="train"):
        """Used in train/validation loop, independent of `forward`
        Args:
            batch: dgl batched graphs
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)
        Returns:
            Loss
        """
        logits, g_logits = self.forward(batch["graph"])
        # node-level targets and mask
        # targets = batch.ndata["target"]
        # train_mask = batch.ndata["mask"]
        # loss = self._compute_loss(logits[train_mask], targets[train_mask])
        # graph-level targets
        g_targets = batch["g_targets"]
        loss = self._compute_loss(g_logits, g_targets)
        self.log("{}_loss".format(prefix), loss, batch_size=g_targets.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class LitGVPMultiStageModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        hparams = [
            "lr",
            "protein_node_in_dim",
            "protein_node_h_dim",
            "protein_edge_in_dim",
            "protein_edge_h_dim",
            "protein_num_layers",
            "ligand_node_in_dim",
            "ligand_node_h_dim",
            "ligand_edge_in_dim",
            "ligand_edge_h_dim",
            "ligand_num_layers",
            "complex_node_in_dim",
            "complex_node_h_dim",
            "complex_edge_in_dim",
            "complex_edge_h_dim",
            "complex_num_layers",
            "drop_rate",
            "residual",
            "seq_embedding",
        ]
        self.save_hyperparameters(*hparams)
        model_kwargs = {key: kwargs[key] for key in hparams if key in kwargs}
        self.model = GVPMultiStageModel(**model_kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model specific args to the base/parent parser.
        Args:
            parent_parser: Base/parent parser
        Returns:
            parent parser with additional model-specific args
        """
        parser = parent_parser.add_argument_group("GVPMultiStageModel")
        parser.add_argument(
            "--protein_node_h_dim",
            type=int,
            nargs="+",
            default=(100, 16),
            help="protein_node_h_dim in GVP",
        )
        parser.add_argument(
            "--protein_edge_h_dim",
            type=int,
            nargs="+",
            default=(32, 1),
            help="protein_edge_h_dim in GVP",
        )
        parser.add_argument(
            "--ligand_node_h_dim",
            type=int,
            nargs="+",
            default=(100, 16),
            help="ligand_node_h_dim in GVP",
        )
        parser.add_argument(
            "--ligand_edge_h_dim",
            type=int,
            nargs="+",
            default=(32, 1),
            help="ligand_edge_h_dim in GVP",
        )
        parser.add_argument(
            "--complex_node_h_dim",
            type=int,
            nargs="+",
            default=(100, 16),
            help="complex_node_h_dim in GVP",
        )
        parser.add_argument(
            "--complex_edge_h_dim",
            type=int,
            nargs="+",
            default=(32, 1),
            help="complex_edge_h_dim in GVP",
        )

        parser.add_argument("--protein_num_layers", type=int, default=3)
        parser.add_argument("--ligand_num_layers", type=int, default=3)
        parser.add_argument("--complex_num_layers", type=int, default=3)

        parser.add_argument("--drop_rate", type=float, default=0.1)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--seq_embedding", action="store_true")
        parser.set_defaults(residual=False, seq_embedding=False)
        return parent_parser

    def _compute_loss(self, logits, targets):
        # binary classification
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        # regression
        loss = F.mse_loss(logits, targets)
        return loss

    def forward(self, g):
        return self.model(g)

    def _step(self, batch, batch_idx, prefix="train"):
        """Used in train/validation loop, independent of `forward`
        Args:
            batch: dgl batched graphs
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)
        Returns:
            Loss
        """
        logits, g_logits = self.forward(batch["protein_graph"], batch["ligand_graph"], batch["complex_graph"])
        # node-level targets and mask
        # targets = batch.ndata["target"]
        # train_mask = batch.ndata["mask"]
        # loss = self._compute_loss(logits[train_mask], targets[train_mask])
        # graph-level targets
        g_targets = batch["g_targets"]
        loss = self._compute_loss(g_logits, g_targets)
        self.log("{}_loss".format(prefix), loss, batch_size=g_targets.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
