from __future__ import annotations

import argparse


DEFAULT_CONFIG_PATH = "./configs/default.json"
DEFAULT_METHOD = "kl_divergence"
DEFAULT_DATASET = "cifar20"
DEFAULT_AUX_DATASET = "cub"
DEFAULT_MODEL = "vit_base_patch16_224"


def setup_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for incremental distillation runs."""
    parser = argparse.ArgumentParser(description="Incremental distillation experiments.")

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Experiment-level JSON overrides.",
    )
    config_group.add_argument(
        "--method-config",
        type=str,
        default=None,
        help="Method-level JSON overrides. Defaults to configs/methods/<method>.json.",
    )

    experiment_group = parser.add_argument_group("Experiment")
    experiment_group.add_argument(
        "--method",
        "--train-strategy",
        "--train_strategy",
        dest="method",
        default=DEFAULT_METHOD,
        help="Method to run: kl_divergence, dkd, mds, ls, self_distillation, se2d.",
    )
    experiment_group.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Dataset family: cifar20, mixed_cifar, digits, domainnet, mixed_domainnet. "
            "Legacy aliases mixed/mixedd and hyphenated forms are also accepted."
        ),
    )
    experiment_group.add_argument(
        "--aux-dataset",
        type=str,
        default=DEFAULT_AUX_DATASET,
        help='Auxiliary dataset used by "mixed_cifar" or "mixed_domainnet" runs.',
    )
    experiment_group.add_argument(
        "--domains-data",
        "--domains_data",
        dest="domains_data",
        nargs="+",
        type=int,
        default=[0, 4],
        help="Domain ids used for the training data loader.",
    )
    experiment_group.add_argument(
        "--domains-teacher",
        "--domains_teacher",
        dest="domains_teacher",
        type=str,
        default="0",
        help="Teacher prefix used to build checkpoint ids.",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Student backbone from timm.",
    )
    model_group.add_argument(
        "--teacher-arch",
        type=str,
        default=DEFAULT_MODEL,
        help="Teacher backbone for non-foundation runs.",
    )
    model_group.add_argument(
        "--use-foundation",
        action="store_true",
        default=False,
        help="Use foundation DomainNet teachers.",
    )

    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--epochs", type=int, default=1)
    training_group.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=64,
    )
    training_group.add_argument("--lr", type=float, default=1e-4)
    training_group.add_argument("--temperature", type=float, default=10.0)
    training_group.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    training_group.add_argument(
        "--device",
        type=str,
        default=None,
        help='Training device. Defaults to "cuda" when available.',
    )

    tracking_group = parser.add_argument_group("Tracking")
    tracking_group.add_argument(
        "--run-name",
        type=str,
        default="experiment",
        help="Prefix used for wandb naming.",
    )
    tracking_group.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Enable wandb sweep mode.",
    )

    return parser
