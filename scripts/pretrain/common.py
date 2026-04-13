from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.model_factory import ModelFactory
from src.utils.paths import ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None) -> torch.device:
    requested = device or "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def parse_domain_group(group_text: str) -> list[int]:
    parts = [part.strip() for part in str(group_text).split("_") if part.strip()]
    if len(parts) < 2:
        raise ValueError(f"Expected at least two domain ids in '{group_text}'.")
    return [int(part) for part in parts]


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _extract_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def run_epoch(model: nn.Module, dataloader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets, *_ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = _extract_logits(model, inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        predictions = torch.argmax(logits, dim=1)
        total += targets.size(0)
        correct += int((predictions == targets).sum().item())

    epoch_loss = running_loss / max(len(dataloader), 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, dataloader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets, *_ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = _extract_logits(model, inputs)
            loss = criterion(logits, targets)

            running_loss += float(loss.item())
            predictions = torch.argmax(logits, dim=1)
            total += targets.size(0)
            correct += int((predictions == targets).sum().item())

    val_loss = running_loss / max(len(dataloader), 1)
    val_acc = 100.0 * correct / max(total, 1)
    return val_loss, val_acc


def train_teacher(
    *,
    label: str,
    model_name: str,
    num_classes: int,
    train_loader,
    val_loader,
    output_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str | None,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    set_seed(seed)
    device_obj = resolve_device(device)
    output_dir = ensure_dir(Path(output_dir))

    model = ModelFactory().create_teacher(model_name, num_classes=num_classes, pretrained=True)
    if hasattr(model, "unfreeze_parameters"):
        model.unfreeze_parameters()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    model = model.to(device_obj)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"[{label}] device={device_obj} model={model_name} params={trainable_parameters:,}/{total_parameters:,} trainable")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }

    best_model_path = output_dir / "best_model.pth"
    final_model_path = output_dir / "final_model.pth"
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device_obj)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "history": history,
            "model_name": model_name,
            "num_classes": num_classes,
        }
        if metadata:
            checkpoint.update(metadata)

        if val_acc >= history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            checkpoint["history"] = history
            torch.save(checkpoint, best_model_path)

        print(
            f"[{label}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}"
        )

    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    final_checkpoint = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": final_val_acc,
        "history": history,
        "model_name": model_name,
        "num_classes": num_classes,
    }
    if metadata:
        final_checkpoint.update(metadata)
    torch.save(final_checkpoint, final_model_path)

    summary = {
        "label": label,
        "model_name": model_name,
        "num_classes": num_classes,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "best_epoch": history["best_epoch"],
        "best_val_acc": history["best_val_acc"],
        "final_val_acc": final_val_acc,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "training_time_sec": time.time() - start_time,
        "output_dir": str(output_dir),
    }
    if metadata:
        summary.update(metadata)

    save_json(output_dir / "training_history.json", history)
    return summary
