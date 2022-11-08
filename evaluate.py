"""
Evaluate a trained pytorch-lightning model on a given dataset.
"""
from train import (
    evaluate_graph_classification,
    get_datasets,
    evaluate_graph_regression,
    MODEL_CONSTRUCTORS,
)

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import argparse
import os
import json
from pprint import pprint


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, classify=False
) -> pl.LightningModule:
    """Load a ptl model from checkpoint path.
    Args:
        checkpoint_path: the path to `lightning_logs/version_x` or
            the .ckpt file itself.
        model_name: should be a key in `MODEL_CONSTRUCTORS`
    """
    if not checkpoint_path.endswith(".ckpt"):
        # find the .ckpt file
        ckpt_file = os.listdir(os.path.join(checkpoint_path, "checkpoints"))[0]
        ckpt_file_path = os.path.join(
            checkpoint_path, "checkpoints", ckpt_file
        )
    else:
        ckpt_file_path = checkpoint_path
    # load the model from checkpoint
    ModelConstructor = MODEL_CONSTRUCTORS[model_name]
    model = ModelConstructor.load_from_checkpoint(
        ckpt_file_path, classify=classify
    )
    return model


def main(args):
    pl.seed_everything(42, workers=True)
    # 1. Load data
    test_dataset = get_datasets(
        name=args.dataset_name,
        input_type=args.input_type,
        data_dir=args.data_dir,
        residue_featurizer_name=args.residue_featurizer_name,
        use_energy_decoder=args.use_energy_decoder,
        data_suffix=args.data_suffix,
        binary_cutoff=args.binary_cutoff,
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
    classify = args.evaluate_type == "classification"
    model = load_model_from_checkpoint(
        args.checkpoint_path,
        args.model_name,
        classify=classify,
    )
    # 4. Evaluate
    if not classify:
        eval_func = evaluate_graph_regression
    else:
        eval_func = evaluate_graph_classification

    scores = eval_func(
        model,
        test_loader,
        model_name=args.model_name,
        use_energy_decoder=args.use_energy_decoder,
        is_hetero=args.is_hetero,
    )
    pprint(scores)
    # save scores to file
    json.dump(
        scores,
        open(
            os.path.join(
                args.checkpoint_path, f"{args.dataset_alias}_scores.json"
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
    parser.add_argument(
        "--evaluate_type",
        type=str,
        help="regression or classification",
        default="regression",
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
        "--data_suffix",
        help="used to distinguish different verions of the same dataset",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--binary_cutoff",
        help="used to convert PDBBind to a binary classification problem",
        type=float,
        default=None,
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
    # featurizer params
    parser.add_argument(
        "--residue_featurizer_name",
        help="name of the residue featurizer",
        type=str,
        default="MACCS",
    )
    parser.add_argument("--use_energy_decoder", action="store_true")
    parser.add_argument("--is_hetero", action="store_true")
    parser.set_defaults(
        use_energy_decoder=False,
        is_hetero=False,
    )
    args = parser.parse_args()

    print("args:", args)
    # evaluate
    main(args)
