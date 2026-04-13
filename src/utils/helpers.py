from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from src.utils.data_utils import (
    create_cifar20_dataset_and_loader,
    create_digits_dataset_and_loader,
    create_domainnet_dataset_and_loader,
    normalize_dataset_name,
)
from src.utils.paths import REPO_ROOT, ensure_dir


RUN_OUTPUT_ROOT = REPO_ROOT / "outputs"
EVAL_IMAGE_SIZE = 224
EVAL_NUM_WORKERS = 4
DATASET_DOMAIN_COUNTS = {
    "cifar20": 5,
    "mixed_cifar": 5,
    "digits": 6,
    "domainnet": 6,
    "mixed_domainnet": 6,
}


@dataclass(frozen=True)
class DomainEvaluationResult:
    domain: int
    accuracy: float


def set_random(seed: int = 1) -> None:
    """Seed all RNGs used by the training stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tminmax(args: dict[str, Any]) -> tuple[int, int]:
    dataset = normalize_dataset_name(args["dataset"])
    if dataset in {"domainnet", "mixed_domainnet", "digits"}:
        return 1, 6
    return 1, 5


def get_exclude_domain(args: dict[str, Any]) -> int:
    return 0


def _standardize_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (logits - mean) / std


def kd_loss_stand(
    logits_student_in: torch.Tensor,
    logits_teacher_in: torch.Tensor,
    temperature: float = 10.0,
) -> torch.Tensor:
    """LS loss: standardize student/teacher logits, then apply KL distillation."""
    standardized_student = _standardize_logits(logits_student_in)
    standardized_teacher = _standardize_logits(logits_teacher_in)
    return (
        F.kl_div(
            F.log_softmax(standardized_student / temperature, dim=1),
            F.softmax(standardized_teacher / temperature, dim=1),
            reduction="batchmean",
        )
        * (temperature**2)
    )


def _cat_target_and_non_target(probabilities: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Collapse a class distribution into [target mass, non-target mass]."""
    target_prob = probabilities[gt_mask].unsqueeze(1)
    non_target_prob = (probabilities * (~gt_mask)).sum(dim=1, keepdim=True)
    return torch.cat([target_prob, non_target_prob], dim=1)


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 10.0,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo-target DKD adaptation for settings without trusted labels.

    When labels are unavailable, the teacher argmax is used as the target class.
    This preserves the DKD decomposition into target-class KD and non-target KD.
    """
    if targets is None:
        targets = teacher_logits.detach().argmax(dim=1)

    gt_mask = F.one_hot(targets, num_classes=student_logits.shape[1]).to(torch.bool)

    student_prob = F.softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)

    student_binary = _cat_target_and_non_target(student_prob, gt_mask)
    teacher_binary = _cat_target_and_non_target(teacher_prob, gt_mask)
    tckd_loss = (
        F.kl_div(
            student_binary.clamp_min(1e-12).log(),
            teacher_binary,
            reduction="batchmean",
        )
        * (temperature**2)
    )

    masked_student = (student_logits / temperature).masked_fill(gt_mask, -1e9)
    masked_teacher = (teacher_logits / temperature).masked_fill(gt_mask, -1e9)
    nckd_loss = (
        F.kl_div(
            F.log_softmax(masked_student, dim=1),
            F.softmax(masked_teacher, dim=1),
            reduction="batchmean",
        )
        * (temperature**2)
    )

    total_loss = float(alpha) * tckd_loss + float(beta) * nckd_loss
    if return_terms:
        return total_loss, tckd_loss, nckd_loss
    return total_loss


def _build_eval_dataloader(args: dict[str, Any], domain_id: int, device: str):
    dataset_name = normalize_dataset_name(args["dataset"])
    loader_kwargs = {
        "selected_domains": [domain_id],
        "batch_size": args["batch_size"],
        "image_size": EVAL_IMAGE_SIZE,
        "train": False,
        "num_workers": EVAL_NUM_WORKERS,
        "pin_memory": device == "cuda",
    }

    if dataset_name in {"mixed_cifar", "cifar20"}:
        return create_cifar20_dataset_and_loader(**loader_kwargs)[1]
    if dataset_name == "digits":
        return create_digits_dataset_and_loader(**loader_kwargs)[1]
    return create_domainnet_dataset_and_loader(**loader_kwargs)[1]


def _compute_accuracy(student, dataloader, device: str) -> float:
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = student(inputs)
            predictions = torch.argmax(logits, dim=1)
            total += targets.size(0)
            correct += (predictions == targets).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0


def _format_evaluation_markdown(results: list[DomainEvaluationResult]) -> str:
    domain_headers = " | ".join(str(result.domain) for result in results)
    accuracies = " | ".join(f"{result.accuracy:.2f}" for result in results)
    markdown = f"| Domain | {domain_headers} |\n"
    markdown += f"|{'---|' * (len(results) + 1)}\n"
    markdown += f"| Accuracy | {accuracies} |\n"
    return markdown


def _safe_wandb_log(payload: dict[str, Any]) -> None:
    try:
        wandb.log(payload)
    except Exception:
        pass


def evaluate_all_domains(
    student,
    domains_teacher=None,
    tag=None,
    training_step: int = 0,
    args: dict[str, Any] | None = None,
    num_classes: int = 20,
) -> None:
    args = args or {}
    device = args.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = normalize_dataset_name(args["dataset"])
    domain_count = DATASET_DOMAIN_COUNTS[dataset_name]

    student.eval()
    results: list[DomainEvaluationResult] = []
    for domain_id in range(domain_count):
        dataloader = _build_eval_dataloader(args, domain_id, device=device)
        accuracy = _compute_accuracy(student, dataloader, device=device)
        results.append(DomainEvaluationResult(domain=domain_id, accuracy=accuracy))
        _safe_wandb_log(
            {
                f"eval/domain_{domain_id}/accuracy": accuracy,
                "training_step": training_step,
            }
        )

    markdown = _format_evaluation_markdown(results)
    ensure_dir(RUN_OUTPUT_ROOT)
    output_path = RUN_OUTPUT_ROOT / f"student_eval_results_{tag}.md"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(f"Student evaluation results saved to {output_path}")
    print(markdown)

    try:
        table = wandb.Table(
            data=[[result.domain, result.accuracy] for result in results],
            columns=["Domain", "Accuracy"],
        )
        wandb.log({"eval_matrix": table, "training_step": training_step})
        wandb.save(str(output_path))
    except Exception:
        pass
