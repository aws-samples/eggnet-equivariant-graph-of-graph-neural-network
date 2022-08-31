import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import GVPModel

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
            self.kernel = self.add_weight(name='kernel',
                                          shape=(3,input_shape[2], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)

            super(Self_Attention, self).build(input_shape)  

        def call(self, x):
            WQ = K.dot(x, self.kernel[0])
            WK = K.dot(x, self.kernel[1])
            WV = K.dot(x, self.kernel[2])

            QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

            QK = QK / (self.output_dim**0.5)

            QK = K.softmax(QK)

            V = K.batch_dot(QK,WV)

            return V

        def compute_output_shape(self, input_shape):

            return (input_shape[0],input_shape[1],self.output_dim)

        def get_config(self):
            config = {
                'output_dim': self.output_dim
            }
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
            print('Start loading model :', model_name)
            model = load_model(model_name,custom_objects={'Self_Attention': Self_Attention})

        def call(self, inputs):
            """
            If model mode is 1, returns prediction label only
            If model mode is not 1, returns prediction label and predicted binding sites
            """
            y = self.model.predict(inputs)
            return y