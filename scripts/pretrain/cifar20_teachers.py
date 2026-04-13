#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common import parse_domain_group, resolve_device, save_json, train_teacher


DEFAULT_GROUPS = ["0_1", "0_2", "0_3", "0_4"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain CIFAR-20 pair teachers.")
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_GROUPS, help="Underscore-separated domain groups, e.g. 0_1 0_2.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--output-root", default="checkpoints/teachers/cifar20")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from src.utils.data_utils import DEFAULT_NUM_CLASSES, create_cifar20_dataset_and_loader

    output_root = Path(args.output_root)
    device = resolve_device(args.device)
    config_payload = vars(args).copy()
    config_payload["pairs"] = list(args.pairs)
    save_json(output_root / "pretraining_config.json", config_payload)

    all_results = {}
    for pair_text in args.pairs:
        domains = parse_domain_group(pair_text)
        label = f"domain_{'_'.join(str(domain_id) for domain_id in domains)}"
        output_dir = output_root / label

        train_dataset, train_loader = create_cifar20_dataset_and_loader(
            selected_domains=domains,
            batch_size=args.batch_size,
            image_size=224,
            train=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_dataset, val_loader = create_cifar20_dataset_and_loader(
            selected_domains=domains,
            batch_size=args.batch_size,
            image_size=224,
            train=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        summary = train_teacher(
            label=label,
            model_name=args.model_name,
            num_classes=DEFAULT_NUM_CLASSES,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            metadata={"domains": domains, "pair": pair_text},
        )
        summary["num_train_samples"] = len(train_dataset)
        summary["num_val_samples"] = len(val_dataset)
        save_json(output_dir / "domain_info.json", summary)
        all_results[label] = summary

    save_json(output_root / "all_domains_results.json", all_results)


if __name__ == "__main__":
    main()
