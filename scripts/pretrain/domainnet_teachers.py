#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common import parse_domain_group, resolve_device, save_json, train_teacher


DEFAULT_PAIRS = ["0_1", "0_2", "0_3", "0_4", "0_5"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain DomainNet pair teachers.")
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="Pair ids such as 0_1 0_2 0_5.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--output-root", default="checkpoints/teachers/domainnet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from src.utils.data_utils import (
        DOMAINNET_NUM_CLASSES,
        DOMAINNET_PAIR_NAMES,
        create_domainnet_dataset_and_loader,
        get_domainnet_domain_names,
    )

    output_root = Path(args.output_root)
    device = resolve_device(args.device)
    available_domains = get_domainnet_domain_names()
    config_payload = vars(args).copy()
    config_payload["pairs"] = list(args.pairs)
    config_payload["resolved_domain_order"] = available_domains
    save_json(output_root / "pretraining_config.json", config_payload)

    all_results = {}
    for pair_text in args.pairs:
        domains = parse_domain_group(pair_text)
        label = f"pair_{pair_text}"
        output_dir = output_root / label

        train_dataset, train_loader = create_domainnet_dataset_and_loader(
            selected_domains=domains,
            batch_size=args.batch_size,
            image_size=224,
            train=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_dataset, val_loader = create_domainnet_dataset_and_loader(
            selected_domains=domains,
            batch_size=args.batch_size,
            image_size=224,
            train=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        resolved_names = [available_domains[domain_id] for domain_id in domains] if available_domains else []
        pair_name = DOMAINNET_PAIR_NAMES.get(label, " + ".join(resolved_names))
        summary = train_teacher(
            label=label,
            model_name=args.model_name,
            num_classes=DOMAINNET_NUM_CLASSES,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            metadata={
                "pair": pair_text,
                "pair_name": pair_name,
                "domains": domains,
                "domain_names": resolved_names,
            },
        )
        summary["num_train_samples"] = len(train_dataset)
        summary["num_val_samples"] = len(val_dataset)
        save_json(output_dir / "pair_info.json", summary)
        all_results[label] = summary

    save_json(output_root / "all_pairs_results.json", all_results)


if __name__ == "__main__":
    main()
