"""
Evaluate a trained pytorch-lightning model on a given dataset.
"""
import pytorch_lightning as pl

import argparse
import os
import json
from pprint import pprint
from torch.utils.data import DataLoader

# custom imports
from ppi.model import LitGVPModel
from train import get_datasets, evaluate_graph_regression, MODEL_CONSTRUCTORS


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str
) -> pl.LightningModule:
    """Load a ptl model from checkpoint path.
    Args:
        checkpoint_path: the path to `lightning_logs/version_x`
        model_name: should be a key in `MODEL_CONSTRUCTORS`
    """
    # find the .ckpt file
    ckpt_file = os.listdir(os.path.join(checkpoint_path, "checkpoints"))[0]
    ckpt_file_path = os.path.join(checkpoint_path, "checkpoints", ckpt_file)
    # load the model from checkpoint
    ModelConstructor = MODEL_CONSTRUCTORS[model_name]
    model = ModelConstructor.load_from_checkpoint(ckpt_file_path)
    return model


def main(args):
    pl.seed_everything(42, workers=True)
    # 1. Load data
    test_dataset = get_datasets(
        name=args.dataset_name,
        input_type=args.input_type,
        data_dir=args.data_dir,
        test_only=True,
    )
    print(
        "Data loaded:",
        len(test_dataset),
    )
    # 2. Prepare data loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )
    # 3. Prepare model
    model = load_model_from_checkpoint(args.checkpoint_path, args.model_name)
    # 4. Evaluate
    scores = evaluate_graph_regression(model, test_loader)
    pprint(scores)
    # save scores to file
    json.dump(
        scores,
        open(
            os.path.join(
                args.checkpoint_path, "{args.dataset_alias}_scores.json"
            ),
            "w",
        ),
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gvp",
        help="Choose from %s" % ", ".join(list(MODEL_CONSTRUCTORS.keys())),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="ptl checkpoint path like `lightning_logs/version_x`",
        required=True,
    )

    # dataset params
    parser.add_argument(
        "--dataset_name",
        help="dataset name",
        type=str,
        default="PepBDB",
    )
    parser.add_argument(
        "--input_type",
        help="data input type",
        type=str,
        default="complex",
    )
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dataset_alias",
        help="Short name for the test dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--bs", type=int, default=64, help="batch size for test data"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers used in DataLoader",
    )

    args = parser.parse_args()

    print("args:", args)
    # evaluate
    main(args)
