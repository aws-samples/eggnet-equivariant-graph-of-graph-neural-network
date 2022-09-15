import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import GVPModel, MultiStageGVPModel


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


class LitMultiStageGVPModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        hparams = [
            "lr",
            "protein_node_in_dim",
            "protein_edge_in_dim",
            "ligand_node_in_dim",
            "ligand_edge_in_dim",
            "complex_edge_in_dim",
            "stage1_node_h_dim",
            "stage1_edge_h_dim",
            "stage2_node_h_dim",
            "stage2_edge_h_dim",
            "stage1_num_layers",
            "stage2_num_layers",
            "drop_rate",
            "residual",
            "seq_embedding",
            "use_energy_decoder",
            "vdw_N",
            "max_vdw_interaction",
            "min_vdw_interaction",
            "dev_vdw_radius",
            "loss_der1_ratio",
            "loss_der2_ratio",
            "min_loss_der2",
        ]
        self.save_hyperparameters(*hparams)
        model_kwargs = {key: kwargs[key] for key in hparams if key in kwargs}
        self.model = MultiStageGVPModel(**model_kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Adds model specific args to the base/parent parser.
        Args:
            parent_parser: Base/parent parser
        Returns:
            parent parser with additional model-specific args
        """
        parser = parent_parser.add_argument_group("MultiStageGVPModel")
        parser.add_argument(
            "--stage1_node_h_dim",
            type=int,
            nargs="+",
            default=(100, 16),
            help="protein_node_h_dim in GVP",
        )
        parser.add_argument(
            "--stage1_edge_h_dim",
            type=int,
            nargs="+",
            default=(32, 1),
            help="protein_edge_h_dim in GVP",
        )
        parser.add_argument(
            "--stage2_node_h_dim",
            type=int,
            nargs="+",
            default=(100, 16),
            help="complex_node_h_dim in GVP",
        )
        parser.add_argument(
            "--stage2_edge_h_dim",
            type=int,
            nargs="+",
            default=(32, 1),
            help="complex_edge_h_dim in GVP",
        )
        parser.add_argument(
            "--no_rotor_penalty",
            action="store_true",
            help="rotor penaly",
        )
        parser.add_argument(
            "--vdw_N",
            help="vdw N",
            type=float,
            default=6.0,
        )
        parser.add_argument(
            "--max_vdw_interaction",
            help="max vdw _interaction",
            type=float,
            default=0.0356,
        )
        parser.add_argument(
            "--min_vdw_interaction",
            help="min vdw _interaction",
            type=float,
            default=0.0178,
        )
        parser.add_argument(
            "--dev_vdw_radius",
            help="deviation of vdw radius",
            type=float,
            default=0.2,
        )
        parser.add_argument(
            "--loss_der1_ratio",
            help="loss der1 ratio",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--loss_der2_ratio",
            help="loss der2 ratio",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--min_loss_der2",
            help="min loss der2",
            type=float,
            default=-20.0,
        )

        parser.add_argument("--stage1_num_layers", type=int, default=3)
        parser.add_argument("--stage2_num_layers", type=int, default=3)

        parser.add_argument("--drop_rate", type=float, default=0.1)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--seq_embedding", action="store_true")
        parser.set_defaults(residual=False, seq_embedding=False)
        return parent_parser

    def _compute_loss(self, logits, targets, loss_der1=0, loss_der2=0):
        # binary classification
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        # regression
        if self.hparams.use_energy_decoder:
            loss_all = 0.0
            loss = F.mse_loss(logits, targets)
            loss_der2 = loss_der2.clamp(min=self.hparams.min_loss_der2)
            loss_all += loss
            loss_all += loss_der1.sum() * self.hparams.loss_der1_ratio
            loss_all += loss_der2.sum() * self.hparams.loss_der2_ratio
            return loss_all
        else:
            loss = F.mse_loss(logits, targets)
            return loss

    def forward(self, protein_graph, ligand_graph, complex_graph, sample=None, cal_der_loss=False, atom_to_residue=None):
        return self.model(protein_graph, ligand_graph, complex_graph, sample=sample, cal_der_loss=cal_der_loss, atom_to_residue=None)

    def _step(self, batch, batch_idx, prefix="train"):
        """Used in train/validation loop, independent of `forward`
        Args:
            batch: dgl batched graphs
            batch_idx: index of current batch
            prefix: Prefix for the loss: XXX_loss (train, validation, test)
        Returns:
            Loss
        """
        if self.hparams.use_energy_decoder:
            cal_der_loss = False
            if prefix == "train":
                if self.hparams.loss_der1_ratio > 0 or self.hparams.loss_der2_ratio > 0.0:
                    cal_der_loss = True
            if self.hparams.is_hetero:
                energies, der1, der2 = self.forward(batch["protein_graph"], batch["ligand_graph"], batch["complex_graph"], batch["sample"], cal_der_loss, batch["atom_to_residue"])
            else:
                energies, der1, der2 = self.forward(batch["protein_graph"], batch["ligand_graph"], batch["complex_graph"], batch["sample"], cal_der_loss)
            g_preds = energies.sum(-1).unsqueeze(-1)
            g_targets = batch["g_targets"]
            loss = self._compute_loss(g_preds, g_targets, der1, der2)
        else:
            logits, g_logits = self.forward(batch["protein_graph"], batch["ligand_graph"], batch["complex_graph"])
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
