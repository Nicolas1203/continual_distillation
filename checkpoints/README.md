# Checkpoint Layout

This directory stores the teacher checkpoints consumed by the training code.

The runtime loader resolves checkpoints from `checkpoints/teachers/` according to
the dataset name and teacher id. The logic lives in
`src/utils/checkpoints.py`.

## Expected Structure

```text
checkpoints/
└── teachers/
    ├── cifar20/
    │   └── domain_<teacher_id>/
    │       └── best_model.pth
    ├── digits/
    │   └── <digits_alias>/
    │       └── best_model.pth
    ├── domainnet/
    │   └── pair_<teacher_id>/
    │       └── best_model.pth
    └── domainnet_foundation/
        └── pair_<teacher_id>/
            ├── best_foundation_model_vit_huge_patch14_clip_224.laion2b.pth
            ├── best_foundation_model*.pth
            └── best_model.pth
```

## Dataset-Specific Naming

### CIFAR-20 and `mixed_cifar`

Teacher ids are resolved under:

```text
checkpoints/teachers/cifar20/domain_<teacher_id>/best_model.pth
```

Example:

```text
teacher_id = 0_1
-> checkpoints/teachers/cifar20/domain_0_1/best_model.pth
```

### DomainNet and `mixed_domainnet`

Teacher ids are resolved under:

```text
checkpoints/teachers/domainnet/pair_<teacher_id>/best_model.pth
```

Example:

```text
teacher_id = 0_3
-> checkpoints/teachers/domainnet/pair_0_3/best_model.pth
```

If `--use-foundation` is enabled, the loader reads from:

```text
checkpoints/teachers/domainnet_foundation/pair_<teacher_id>/
```

and prefers:

1. `best_foundation_model_vit_huge_patch14_clip_224.laion2b.pth`
2. any `best_foundation_model*.pth`
3. `best_model.pth`

### digits

Digits checkpoints use alias directories rather than raw pair ids.

```text
0_1 -> mnist_svhn
0_2 -> mnist_mnist-m
0_3 -> mnist_usps
0_4 -> mnist_kmnist
0_5 -> mnist_fashion-mnist
```

Example:

```text
teacher_id = 0_2
-> checkpoints/teachers/digits/mnist_mnist-m/best_model.pth
```

## Checkpoint Contents

The loader accepts either:

- a raw state dict, or
- a dictionary containing `model_state_dict`

The pretraining scripts write checkpoints with at least:

- `model_state_dict`
- `model_name`
- `num_classes`
- optimizer and training metadata

## Pretraining Scripts

The standard teacher pretraining scripts populate the non-foundation tree:

- `scripts/pretrain/cifar20_teachers.py`
- `scripts/pretrain/digits_teachers.py`
- `scripts/pretrain/domainnet_teachers.py`

These scripts are expected to produce the folder names shown above.

## Notes

- Student checkpoints and evaluation outputs do not belong here; they are written under `outputs/`.
- If a teacher checkpoint is not found, check the dataset name, `domains_teacher`, and the implied task/domain id first.
