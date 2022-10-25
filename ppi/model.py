import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .modules import GVPModel, MultiStageGVPModel
from ppi.data_utils import get_residue_featurizer


def infer_input_dim(g: dgl.DGLGraph) -> tuple:
    """Infer node_in_dim and edge_in_dim for GVPModel"""
    node_in_dim = (
        g.ndata["node_s"].shape[1],
        g.ndata["node_v"].shape[1],
    )
    edge_in_dim = (
        g.edata["edge_s"].shape[1],
        g.edata["edge_v"].shape[1],
    )
    return node_in_dim, edge_in_dim


class LitGVPModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get("g", None):
            node_in_dim, edge_in_dim = infer_input_dim(kwargs["g"])
            kwargs["node_in_dim"] = node_in_dim
            kwargs["edge_in_dim"] = edge_in_dim

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
            "use_energy_decoder",
            "intra_mol_energy",
            "energy_agg_type",
            "vdw_N",
            "max_vdw_interaction",
            "min_vdw_interaction",
            "dev_vdw_radius",
            "loss_der1_ratio",
            "loss_der2_ratio",
            "min_loss_der2",
        ]

        model_kwargs = {key: kwargs[key] for key in hparams if key in kwargs}
        self.model = GVPModel(**model_kwargs)
        self.save_hyperparameters(*hparams)

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
        parser.add_argument(
            "--energy_agg_type",
            help="type of energy aggregator in the format of {agg_type}_{bn:0/1}",
            type=str,
            default="0_1",
        )
        parser.set_defaults(residual=False, seq_embedding=False)
        return parent_parser

    def _compute_loss(self, logits, targets, loss_der1=0, loss_der2=0):
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

    def forward(self, batch):
        return self.model(batch["graph"], sample=batch.get("sample"))

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
                if (
                    self.hparams.loss_der1_ratio > 0
                    or self.hparams.loss_der2_ratio > 0.0
                ):
                    cal_der_loss = True

            energies, der1, der2 = self.model.forward(
                batch["graph"],
                sample=batch["sample"],
                cal_der_loss=cal_der_loss,
            )
            g_preds = energies.sum(-1).unsqueeze(-1)
            g_targets = batch["g_targets"]
            loss = self._compute_loss(g_preds, g_targets, der1, der2)
        else:
            logits, g_logits = self.model.forward(
                batch["graph"],
            )
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


class LitHGVPModel(pl.LightningModule):
    """
    End-to-end hierachical GVP model.
    Input: a pair of molecules
    Output: a scalar
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.residue_featurizer = get_residue_featurizer(
            kwargs["residue_featurizer_name"]
        )
        # lazy init for model that requires an input datum
        if kwargs.get("g", None):
            node_in_dim, edge_in_dim = infer_input_dim(kwargs["g"])
            node_in_dim = (
                node_in_dim[0] + self.residue_featurizer.output_size,
                node_in_dim[1],
            )
            kwargs["node_in_dim"] = node_in_dim
            kwargs["edge_in_dim"] = edge_in_dim

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
            "residue_featurizer_name",
            "use_energy_decoder",
            "intra_mol_energy",
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
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--drop_rate", type=float, default=0.1)
        parser.add_argument("--residual", action="store_true")
        parser.add_argument("--seq_embedding", action="store_true")
        parser.set_defaults(residual=False, seq_embedding=False)
        return parent_parser

    def _compute_loss(self, logits, targets, loss_der1=0, loss_der2=0):
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

    def forward(self, batch, cal_der_loss=False):
        bg, smiles_strings = batch["graph"], batch["smiles_strings"]
        node_s = bg.ndata["node_s"]
        residue_embeddings = self.residue_featurizer(smiles_strings, device=self.device)
        bg.ndata["node_s"] = torch.cat((node_s, residue_embeddings), axis=1)
        if self.hparams.use_energy_decoder:
            return self.model(
                bg, sample=batch.get("sample"), cal_der_loss=cal_der_loss
            )
        else:
            return self.model(bg, sample=batch.get("sample"))

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
                if (
                    self.hparams.loss_der1_ratio > 0
                    or self.hparams.loss_der2_ratio > 0.0
                ):
                    cal_der_loss = True

            energies, der1, der2 = self.forward(
                batch,
                cal_der_loss=cal_der_loss,
            )
            g_preds = energies.sum(-1).unsqueeze(-1)
            g_targets = batch["g_targets"]
            loss = self._compute_loss(g_preds, g_targets, der1, der2)
        else:
            logits, g_logits = self.forward(batch)
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


try:
    from keras.models import load_model
    import tensorflow as tf

    from keras.preprocessing import sequence
    from keras import backend as K
    from keras.engine.topology import Layer
except ModuleNotFoundError:
    pass
else:

    class Self_Attention(Layer):
        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super(Self_Attention, self).__init__()

        def build(self, input_shape):
            self.kernel = self.add_weight(
                name="kernel",
                shape=(3, input_shape[2], self.output_dim),
                initializer="uniform",
                trainable=True,
            )

            super(Self_Attention, self).build(input_shape)

        def call(self, x):
            WQ = K.dot(x, self.kernel[0])
            WK = K.dot(x, self.kernel[1])
            WV = K.dot(x, self.kernel[2])

            QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

            QK = QK / (self.output_dim ** 0.5)

            QK = K.softmax(QK)

            V = K.batch_dot(QK, WV)

            return V

        def compute_output_shape(self, input_shape):

            return (input_shape[0], input_shape[1], self.output_dim)

        def get_config(self):
            config = {"output_dim": self.output_dim}
            base_config = super(Self_Attention, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class CAMPModel(tf.keras.Model):
        """
        CAMP model modified from https://github.com/twopin/CAMP

        model_mode: 1 (to get affinity value), otherwise get affinity value + predicted binding sites
        model_name: path to the CAMP model, options [efs/data/CAMP/models/CAMP.h5, efs/data/CAMP/models/CAMP_BS.h5]
        """

        def __init__(self, model_mode, model_name):
            super().__init__()
            # model_name='./model/CAMP.h5' # Update to point to model directory
            print("Start loading model :", model_name)
            model = load_model(
                model_name, custom_objects={"Self_Attention": Self_Attention}
            )

        def call(self, inputs):
            """
            If model mode is 1, returns prediction label only
            If model mode is not 1, returns prediction label and predicted binding sites
            """
            y = self.model.predict(inputs)
            return y


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
            "residue_featurizer_name",
            "use_energy_decoder",
            "is_hetero",
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

    def forward(
        self,
        protein_graph,
        ligand_graph,
        complex_graph,
        sample=None,
        cal_der_loss=False,
        atom_to_residue=None,
    ):
        return self.model(
            protein_graph,
            ligand_graph,
            complex_graph,
            sample=sample,
            cal_der_loss=cal_der_loss,
            atom_to_residue=atom_to_residue,
        )

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
                if (
                    self.hparams.loss_der1_ratio > 0
                    or self.hparams.loss_der2_ratio > 0.0
                ):
                    cal_der_loss = True
            if self.hparams.is_hetero:
                energies, der1, der2 = self.forward(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    cal_der_loss,
                    batch["atom_to_residue"],
                )
            else:
                energies, der1, der2 = self.forward(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    cal_der_loss,
                )
            g_preds = energies.sum(-1).unsqueeze(-1)
            g_targets = batch["g_targets"]
            loss = self._compute_loss(g_preds, g_targets, der1, der2)
        else:
            logits, g_logits = self.forward(
                batch["protein_graph"],
                batch["ligand_graph"],
                batch["complex_graph"],
            )
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


class LitMultiStageHGVPModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.residue_featurizer = get_residue_featurizer(
            kwargs["residue_featurizer_name"], device=self.device
        )
        # lazy init for model that requires an input datum
        if kwargs.get("g", None):
            node_in_dim, edge_in_dim = infer_input_dim(
                kwargs["g"]
            )
            node_in_dim = (
                node_in_dim[0] + self.residue_featurizer.output_size,
                node_in_dim[1],
            )
            kwargs["protein_node_in_dim"] = node_in_dim
            kwargs["protein_edge_in_dim"] = edge_in_dim
            kwargs["ligand_node_in_dim"] = node_in_dim
            kwargs["ligand_edge_in_dim"] = edge_in_dim

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
            "residue_featurizer_name",
            "use_energy_decoder",
            "is_hetero",
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

    def forward(
        self,
        protein_graph,
        ligand_graph,
        complex_graph,
        sample=None,
        smiles_strings=None,
        cal_der_loss=False,
        atom_to_residue=None,
        ligand_smiles=None
    ):
        protein_node_s = protein_graph.ndata["node_s"]
        residue_embeddings, _ = self.residue_featurizer(smiles_strings, device=self.device)
        protein_graph.ndata["node_s"] = torch.cat(
            (protein_node_s, residue_embeddings), axis=1
        )
        _, atom_embeddings = self.residue_featurizer(ligand_smiles, device=self.device)
        ligand_node_s = ligand_graph.ndata["node_s"]
        ligand_graph.ndata["node_s"] = torch.cat(
            (ligand_node_s, atom_embeddings), axis=1
        )
        return self.model(
            protein_graph,
            ligand_graph,
            complex_graph,
            sample=sample,
            cal_der_loss=cal_der_loss,
            atom_to_residue=atom_to_residue,
        )

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
                if (
                    self.hparams.loss_der1_ratio > 0
                    or self.hparams.loss_der2_ratio > 0.0
                ):
                    cal_der_loss = True
            if self.hparams.is_hetero:
                energies, der1, der2 = self.forward(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    batch["smiles_strings"],
                    cal_der_loss,
                    batch["atom_to_residue"],
                    batch["ligand_smiles"]
                )
            else:
                energies, der1, der2 = self.forward(
                    batch["protein_graph"],
                    batch["ligand_graph"],
                    batch["complex_graph"],
                    batch["sample"],
                    batch["smiles_strings"],
                    cal_der_loss,
                    batch["ligand_smiles"]
                )
            g_preds = energies.sum(-1).unsqueeze(-1)
            g_targets = batch["g_targets"]
            loss = self._compute_loss(g_preds, g_targets, der1, der2)
        else:
            logits, g_logits = self.forward(
                batch["protein_graph"],
                batch["ligand_graph"],
                batch["complex_graph"],
                smiles_strings=batch["smiles_strings"],
                ligand_smiles = batch["ligand_smiles"]
            )
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
