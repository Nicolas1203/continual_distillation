"""Training loops for the maintained distillation methods.

The code here deliberately favors readable, explicit method loops over deeply
generic abstractions. Small helpers handle shared setup, logging, and
checkpointing, while each method keeps its own core optimization logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import wandb

from src.models.model_factory import ModelFactory
from src.models.vision_models import VisionTeacher
from src.utils.checkpoints import load_teacher_checkpoint
from src.utils.data_utils import normalize_dataset_name
from src.utils.helpers import dkd_loss, get_exclude_domain, kd_loss_stand
from src.utils.paths import REPO_ROOT, ensure_dir


DEFAULT_NUM_CLASSES = 20
RUN_OUTPUT_ROOT = REPO_ROOT / "outputs"
EMBED_DIM_TO_VIT_MODEL = {
    192: "vit_tiny_patch16_224",
    384: "vit_small_patch16_224",
    768: "vit_base_patch16_224",
    1024: "vit_large_patch16_224",
}


# ---------------------------------------------------------------------------
# Logging and checkpoint helpers
# ---------------------------------------------------------------------------

def _to_python_indices(indexes: Iterable) -> list[int]:
    return [int(index.item()) if isinstance(index, torch.Tensor) else int(index) for index in indexes]


def _update_run_metadata(domains_teacher, domains_data, epochs: int) -> None:
    try:
        wandb.config.update(
            {
                "domains_teacher": domains_teacher,
                "domains_data": domains_data,
                "epochs": epochs,
            }
        )
    except Exception:
        pass


def _safe_log_metrics(metrics: dict[str, Any]) -> None:
    try:
        wandb.log(metrics)
    except Exception:
        pass


def _log_epoch_metrics(
    domains_teacher,
    epoch: int,
    running_loss: float,
    num_batches: int,
    teacher_accuracy: float | None = None,
) -> None:
    metrics = {
        "train/loss": running_loss / max(num_batches, 1),
        "train/epoch": epoch,
        "train/domains_teacher": domains_teacher,
    }
    if teacher_accuracy is not None:
        metrics["train/teacher_accuracy"] = teacher_accuracy
    _safe_log_metrics(metrics)


def _save_student_checkpoint(student, domains_teacher, domains_data, filename_prefix: str) -> None:
    ensure_dir(RUN_OUTPUT_ROOT)
    domains_str = "_".join(str(domain) for domain in domains_data)
    save_path = RUN_OUTPUT_ROOT / f"{filename_prefix}student_teacher{domains_teacher}_domains_{domains_str}.pth"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "domains_teacher": domains_teacher,
            "domains_data": domains_data,
        },
        save_path,
    )
    print(f"Student model saved to {save_path}")


def _clone_student(student, args: dict[str, Any], num_classes: int, device: str):
    cloned_student = ModelFactory().create_student(
        args["model"],
        num_classes=num_classes,
        pretrained=True,
    )
    cloned_student.load_state_dict(student.state_dict(), strict=False)
    cloned_student.to(device)
    cloned_student.eval()
    return cloned_student


# ---------------------------------------------------------------------------
# Teacher / student setup
# ---------------------------------------------------------------------------

def _infer_teacher_arch_from_checkpoint(checkpoint: dict[str, Any], fallback_model: str) -> str:
    model_name = checkpoint.get("model_name")
    if model_name:
        return model_name

    state_dict = checkpoint.get("model_state_dict", {})
    patch_weight = state_dict.get("model.patch_embed.proj.weight")
    if patch_weight is None:
        patch_weight = state_dict.get("patch_embed.proj.weight")
    if patch_weight is None or len(getattr(patch_weight, "shape", ())) < 3:
        return fallback_model

    embed_dim = int(patch_weight.shape[0])
    return EMBED_DIM_TO_VIT_MODEL.get(embed_dim, fallback_model)


def create_teacher_model(
    dataset: str,
    args: dict[str, Any],
    num_classes: int,
    use_foundation: bool = True,
    model_name: str | None = None,
):
    """Build the teacher architecture expected by the loaded checkpoint."""
    normalized_dataset = normalize_dataset_name(dataset)
    if normalized_dataset in {"domainnet", "mixed_domainnet"} and use_foundation:
        return timm.create_model(
            "vit_huge_patch14_clip_224.laion2b",
            pretrained=False,
            num_classes=num_classes,
        )

    teacher_arch = model_name or args["teacher_arch"]
    return VisionTeacher(model_name=teacher_arch, num_classes=num_classes, pretrained=False)


def _load_teacher_model(args: dict[str, Any], domains_teacher, num_classes: int, device: str):
    use_foundation = args.get("use_foundation", False)
    checkpoint = load_teacher_checkpoint(
        domains_teacher,
        args=args,
        use_foundation=use_foundation,
    )
    teacher_arch = _infer_teacher_arch_from_checkpoint(checkpoint, args["teacher_arch"])
    teacher = create_teacher_model(
        args["dataset"],
        args,
        num_classes,
        use_foundation=use_foundation,
        model_name=teacher_arch,
    )
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=False)
    teacher.to(device)
    teacher.eval()
    return teacher


def _prepare_student_and_teacher(
    student,
    args: dict[str, Any],
    domains_teacher,
    domains_data,
    num_classes: int,
    device: str,
    epochs: int,
):
    teacher = _load_teacher_model(args, domains_teacher, num_classes, device)
    student.to(device)
    student.train()
    _update_run_metadata(domains_teacher, domains_data, epochs)
    return teacher


# ---------------------------------------------------------------------------
# Shared losses and small utilities
# ---------------------------------------------------------------------------

def _kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * (temperature**2)
    )


def _coerce_domains_data(domains_data) -> list[int]:
    return list(domains_data or [0])


def _to_device_tensor(domain_id, device: str) -> torch.Tensor:
    if isinstance(domain_id, torch.Tensor):
        return domain_id.to(device)
    return torch.tensor(domain_id, device=device)


def train_kl_divergence(
    student,
    optimizer,
    domains_teacher="0",
    domains_data=None,
    device="cuda",
    epochs=1,
    args=None,
    dataloader=None,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """Train the student against the current teacher with plain KL distillation."""
    args = args or {}
    domains_data = _coerce_domains_data(domains_data)
    teacher = _prepare_student_and_teacher(
        student,
        args,
        domains_teacher,
        domains_data,
        num_classes,
        device,
        epochs,
    )

    temperature = args["temperature"]
    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        teacher_correct = 0
        teacher_total = 0

        for inputs, targets, _, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_predictions = torch.argmax(teacher_logits, dim=1)
                teacher_correct += (teacher_predictions == targets).sum().item()
                teacher_total += targets.size(0)

            student_logits = student(inputs)
            loss = _kl_distillation_loss(student_logits, teacher_logits, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        teacher_accuracy = 100.0 * teacher_correct / teacher_total if teacher_total > 0 else 0.0
        _log_epoch_metrics(
            domains_teacher,
            epoch,
            running_loss,
            num_batches,
            teacher_accuracy=teacher_accuracy,
        )

    _save_student_checkpoint(student, domains_teacher, domains_data, filename_prefix="REBUTTAL_")
    return student


def train_ls(
    student,
    optimizer,
    domains_teacher="0",
    domains_data=None,
    device="cuda",
    epochs=1,
    args=None,
    dataloader=None,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """Train the LS variant with pure standardized-logit distillation."""
    args = args or {}
    domains_data = _coerce_domains_data(domains_data)
    teacher = _prepare_student_and_teacher(
        student,
        args,
        domains_teacher,
        domains_data,
        num_classes,
        device,
        epochs,
    )

    loss_temperature = args.get("ls_temperature", 10.0)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        for inputs, _, _, _ in dataloader:
            inputs = inputs.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss = kd_loss_stand(student_logits, teacher_logits, temperature=loss_temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        _log_epoch_metrics(domains_teacher, epoch, running_loss, num_batches)

    _save_student_checkpoint(student, domains_teacher, domains_data, filename_prefix="")
    return student


def train_dkd(
    student,
    optimizer,
    domains_teacher="0",
    domains_data=None,
    device="cuda",
    epochs=1,
    args=None,
    dataloader=None,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """Train the pseudo-target DKD adaptation used in this repository."""
    args = args or {}
    domains_data = _coerce_domains_data(domains_data)
    teacher = _prepare_student_and_teacher(
        student,
        args,
        domains_teacher,
        domains_data,
        num_classes,
        device,
        epochs,
    )

    dkd_alpha = args.get("dkd_alpha", 1.0)
    dkd_beta = args.get("dkd_beta", 8.0)
    dkd_temperature = args.get("dkd_temperature", 10.0)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        for inputs, _, _, _ in dataloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                pseudo_targets = teacher_logits.argmax(dim=1)

            student_logits = student(inputs)
            loss = dkd_loss(
                student_logits,
                teacher_logits,
                targets=pseudo_targets,
                alpha=dkd_alpha,
                beta=dkd_beta,
                temperature=dkd_temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        _log_epoch_metrics(domains_teacher, epoch, running_loss, num_batches)

    _save_student_checkpoint(student, domains_teacher, domains_data, filename_prefix="")
    return student


def _select_mds_indices(teacher, dataloader, device: str, selection_ratio_pct: int) -> set[int]:
    """Pick the middle-entropy samples used by the MDS baseline."""
    scores = np.zeros(len(dataloader.dataset))
    processed_indices: list[int] = []

    for inputs, _, _, indexes in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
            probabilities = F.softmax(teacher_logits, dim=-1)
            entropies = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1)

        batch_indices = _to_python_indices(indexes)
        scores[batch_indices] = entropies.detach().cpu().numpy()
        processed_indices.extend(batch_indices)

    processed_indices_array = np.array(processed_indices)
    subset_scores = scores[processed_indices_array]
    num_to_select = int(len(processed_indices_array) * (selection_ratio_pct / 100))

    sort_indices = np.argsort(subset_scores)
    middle_index = len(sort_indices) // 2
    low_count = num_to_select // 2
    high_count = num_to_select - low_count
    selected_meta_indices = sort_indices[max(0, middle_index - low_count) : middle_index + high_count]
    return set(processed_indices_array[selected_meta_indices].tolist())


def train_mds(
    student,
    optimizer,
    domains_teacher="0",
    domains_data=None,
    device="cuda",
    epochs=1,
    args=None,
    dataloader=None,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """Train on the subset of samples selected by teacher entropy."""
    args = args or {}
    domains_data = _coerce_domains_data(domains_data)
    teacher = _prepare_student_and_teacher(
        student,
        args,
        domains_teacher,
        domains_data,
        num_classes,
        device,
        epochs,
    )

    selected_sample_ids = _select_mds_indices(
        teacher,
        dataloader,
        device=device,
        selection_ratio_pct=int(args.get("selection_ratio_pct", 50)),
    )

    print("total number of training samples:", len(selected_sample_ids))
    print("sum of index:", sum(selected_sample_ids))

    temperature = args["temperature"]
    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        for inputs, _, _, indexes in dataloader:
            inputs = inputs.to(device)
            batch_indices = _to_python_indices(indexes)
            batch_mask = torch.tensor(
                [index in selected_sample_ids for index in batch_indices],
                device=device,
                dtype=torch.bool,
            )
            if not batch_mask.any():
                continue

            with torch.no_grad():
                teacher_logits = teacher(inputs)[batch_mask]
            student_logits = student(inputs)[batch_mask]
            loss = _kl_distillation_loss(student_logits, teacher_logits, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        _log_epoch_metrics(domains_teacher, epoch, running_loss, num_batches)

    _save_student_checkpoint(student, domains_teacher, domains_data, filename_prefix="")
    return student


def _train_with_old_student(
    student,
    optimizer,
    domains_teacher,
    domains_data,
    device,
    epochs,
    args,
    old_student,
    dataloader,
    num_classes,
    *,
    domain_unknown: bool,
):
    teacher = _prepare_student_and_teacher(
        student,
        args,
        domains_teacher,
        domains_data,
        num_classes,
        device,
        epochs,
    )

    temperature = args["temperature"]
    old_student_weight = args.get("old_student_weight", 1.0)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        teacher_correct = 0
        teacher_total = 0

        for inputs, targets, domain_id, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            domain_id = _to_device_tensor(domain_id, device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_predictions = torch.argmax(teacher_logits, dim=1)
                teacher_correct += (teacher_predictions == targets).sum().item()
                teacher_total += targets.size(0)

            student_logits = student(inputs)
            teacher_loss = _kl_distillation_loss(student_logits, teacher_logits, temperature)
            loss = teacher_loss

            if old_student is not None:
                with torch.no_grad():
                    old_teacher_logits = old_student(inputs)

                if domain_unknown:
                    old_student_loss = _kl_distillation_loss(student_logits, old_teacher_logits, temperature)
                    loss = teacher_loss + old_student_weight * old_student_loss
                else:
                    exclude_domain = get_exclude_domain(args)
                    mask_old = domain_id != exclude_domain
                    if mask_old.any():
                        old_student_loss = _kl_distillation_loss(
                            student_logits[mask_old],
                            old_teacher_logits[mask_old],
                            temperature,
                        )
                        loss = teacher_loss + old_student_weight * old_student_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        teacher_accuracy = 100.0 * teacher_correct / teacher_total if teacher_total > 0 else 0.0
        _log_epoch_metrics(
            domains_teacher,
            epoch,
            running_loss,
            num_batches,
            teacher_accuracy=teacher_accuracy,
        )

    _save_student_checkpoint(student, domains_teacher, domains_data, filename_prefix="REBUTTAL_")
    return _clone_student(student, args, num_classes, device)


def train_self_distillation(
    student,
    optimizer,
    domains_teacher="0",
    domains_data=None,
    device="cuda",
    epochs=1,
    args=None,
    old_student=None,
    dataloader=None,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """Train the self-distillation variant that reuses the previous student on all samples."""
    args = args or {}
    domains_data = _coerce_domains_data(domains_data)
    return _train_with_old_student(
        student=student,
        optimizer=optimizer,
        domains_teacher=domains_teacher,
        domains_data=domains_data,
        device=device,
        epochs=epochs,
        args=args,
        old_student=old_student,
        dataloader=dataloader,
        num_classes=num_classes,
        domain_unknown=True,
    )


def train_se2d(
    student,
    optimizer,
    domains_teacher="0",
    domains_data=None,
    device="cuda",
    epochs=1,
    args=None,
    old_student=None,
    dataloader=None,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """Train the SE2D variant that only reuses the previous student on retained domains."""
    args = args or {}
    domains_data = _coerce_domains_data(domains_data)
    return _train_with_old_student(
        student=student,
        optimizer=optimizer,
        domains_teacher=domains_teacher,
        domains_data=domains_data,
        device=device,
        epochs=epochs,
        args=args,
        old_student=old_student,
        dataloader=dataloader,
        num_classes=num_classes,
        domain_unknown=False,  # Give External data information.
    )
