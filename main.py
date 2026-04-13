from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
import wandb

from configs.parser import setup_parser
from src.methods import create_method, normalize_method_name
from src.models.model_factory import ModelFactory
from src.utils.data_utils import create_dataset_and_loader, get_num_classes, normalize_dataset_name
from src.utils.helpers import evaluate_all_domains, get_tminmax, set_random
from src.utils.paths import CONFIG_ROOT


WANDB_PROJECT = "continual_distillation"
METHODS_WITH_PREVIOUS_STUDENT = {"self_distillation", "se2d"}


def load_json_config(config_path: str | None) -> dict[str, Any]:
    """Load a JSON configuration file when it exists."""
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_explicit_cli_dests(parser, argv: list[str]) -> set[str]:
    """Return parser destinations that were explicitly provided on the CLI."""
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_destinations: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            continue

        option = token.split("=", 1)[0]
        destination = option_to_dest.get(option)
        if destination is not None:
            explicit_destinations.add(destination)

    return explicit_destinations


def resolve_device(args: dict[str, Any]) -> str:
    """Resolve the runtime device, defaulting to CUDA when available."""
    if args.get("device"):
        return str(args["device"])
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_domain_sequence(args: dict[str, Any]) -> list[int]:
    """Build the sequential teacher-domain schedule for the current run."""
    t_min, t_max = get_tminmax(args)
    domain_sequence = list(range(t_min, t_max))

    # Historical behavior: when external domains are provided, training stops
    # before the first non-zero external domain.
    external_domains = sorted(int(domain_id) for domain_id in args["domains_data"] if int(domain_id) > 0)
    if external_domains:
        cutoff_domain = external_domains[0]
        domain_sequence = [domain_id for domain_id in domain_sequence if domain_id < cutoff_domain]

    return domain_sequence


def resolve_data_domains(args: dict[str, Any]) -> list[int]:
    """Normalize the training-domain list to a plain list of integers."""
    return [int(domain_id) for domain_id in args["domains_data"]]


def build_teacher_id(args: dict[str, Any], domain_id: int) -> str:
    """Compose the checkpoint identifier expected by the teacher loader."""
    base_teacher = str(args["domains_teacher"]).replace("teacher:", "")
    dataset_name = normalize_dataset_name(args["dataset"])
    if dataset_name in {"domainnet", "mixed_domainnet"} and base_teacher.startswith("pair_"):
        base_teacher = base_teacher[len("pair_") :]
    if dataset_name in {"cifar20", "mixed_cifar"} and base_teacher.startswith("domain_"):
        base_teacher = base_teacher[len("domain_") :]
    return f"{base_teacher}_{domain_id}"


def init_wandb(args: dict[str, Any]) -> str:
    """Initialize Weights & Biases and return the run identifier."""
    run_id = f"{args['run_name']}_seed{args['seed']}"
    if args.get("sweep"):
        wandb.init(project=WANDB_PROJECT)
        for key in wandb.config.keys():
            args[key] = wandb.config[key]
        if wandb.run is not None:
            wandb.run.name = f"{args['method']}_{args['epochs']}_{args['batch_size']}_{args['seed']}"
        return run_id

    wandb.init(project=WANDB_PROJECT, config=args)
    if wandb.run is not None:
        wandb.run.name = run_id
    return run_id


def build_effective_args(parser) -> dict[str, Any]:
    """Merge CLI arguments with JSON config overrides while preserving explicit CLI flags."""
    parsed_args = vars(parser.parse_args())
    file_config = load_json_config(parsed_args.get("config_path"))

    merged_args = dict(parsed_args)
    merged_args.update(file_config)

    for destination in get_explicit_cli_dests(parser, sys.argv[1:]):
        merged_args[destination] = parsed_args[destination]

    merged_args["method"] = normalize_method_name(merged_args["method"])
    merged_args["dataset"] = normalize_dataset_name(merged_args["dataset"])
    if not merged_args.get("method_config"):
        merged_args["method_config"] = str(CONFIG_ROOT / "methods" / f"{merged_args['method']}.json")
    merged_args["device"] = resolve_device(merged_args)

    return merged_args


def create_student_and_optimizer(args: dict[str, Any], num_classes: int):
    """Instantiate the student model and optimizer for the run."""
    model_factory = ModelFactory()
    student = model_factory.create_student(args["model"], num_classes=num_classes, pretrained=True)
    student.to(args["device"])
    optimizer = optim.Adam(student.parameters(), lr=args["lr"])
    return student, optimizer


def evaluate_student_after_task(
    student,
    args: dict[str, Any],
    teacher_id: str,
    domain_id: int,
    num_classes: int,
) -> None:
    """Run the standard evaluation pass and persist the markdown summary."""
    domains_str = "_".join(str(domain) for domain in args["domains_data"])
    eval_tag = f"{args['method']}_teacher_{teacher_id}_domains_{domains_str}"
    evaluate_all_domains(
        student,
        domains_teacher=teacher_id,
        tag=eval_tag,
        training_step=domain_id,
        args=args,
        num_classes=num_classes,
    )


def main() -> None:
    parser = setup_parser()
    args = build_effective_args(parser)

    # W&B initialization may consume RNG state, so reseed afterwards to keep
    # training runs deterministic.
    set_random(args["seed"])
    init_wandb(args)
    set_random(args["seed"])

    num_classes = get_num_classes(args)
    student, optimizer = create_student_and_optimizer(args, num_classes=num_classes)

    domain_sequence = resolve_domain_sequence(args)
    args["domains_data"] = resolve_data_domains(args)

    print(f"Sequential Domain Sequence: {domain_sequence}")
    print(f"Training data domains: {args['domains_data']}")
    print(f"Method: {args['method']}")

    previous_student = None
    method = create_method(
        args["method"],
        args=args,
        optimizer=optimizer,
        config_path=args["method_config"],
    )

    for domain_id in domain_sequence:
        teacher_id = build_teacher_id(args, domain_id)
        _, dataloader = create_dataset_and_loader(
            args,
            selected_domains=args["domains_data"],
            train=True,
        )

        train_kwargs = {
            "student": student,
            "domains_teacher": teacher_id,
            "domains_data": args["domains_data"],
            "device": args["device"],
            "epochs": args["epochs"],
            "dataloader": dataloader,
            "num_classes": num_classes,
        }
        if args["method"] in METHODS_WITH_PREVIOUS_STUDENT:
            train_kwargs["old_student"] = previous_student

        result = method.train(
            **train_kwargs,
        )

        if args["method"] in METHODS_WITH_PREVIOUS_STUDENT:
            previous_student = result

        evaluate_student_after_task(
            student=student,
            args=args,
            teacher_id=teacher_id,
            domain_id=domain_id,
            num_classes=num_classes,
        )


if __name__ == "__main__":
    main()
