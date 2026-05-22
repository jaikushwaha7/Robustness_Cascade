import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from models import SmallCNN, get_resnet18

try:
    from torchvision.transforms import AugMix as TorchvisionAugMix
except ImportError:  # pragma: no cover - optional dependency
    TorchvisionAugMix = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover - optional dependency
    EmissionsTracker = None


def _log_wandb(wandb_run: Any, data: dict[str, Any], step: Optional[int] = None) -> None:
    if wandb_run is not None:
        wandb_run.log(data, step=step)


def _start_codecarbon_tracker(enabled: bool, output_dir: str, project_name: str) -> Any:
    if not enabled or EmissionsTracker is None:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        save_to_file=True,
        save_to_api=False,
        save_to_logger=False,
    )
    tracker.start()
    return tracker


def _stop_codecarbon_tracker(tracker: Any, wandb_run: Any, metric_prefix: str) -> Optional[float]:
    if tracker is None:
        return None

    emissions_kg = tracker.stop()
    print(f"[{metric_prefix}] CodeCarbon emissions: {emissions_kg:.6f} kg CO2e")
    if wandb_run is not None:
        wandb_run.log({f"{metric_prefix}/emissions_kg": emissions_kg})
        wandb_run.summary[f"{metric_prefix}/emissions_kg"] = float(emissions_kg)
    return emissions_kg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


def build_clean_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def build_augmix_transform(severity: int = 3,
                           mixture_width: int = 3,
                           chain_depth: int = -1,
                           alpha: float = 1.0) -> transforms.Compose:
    if TorchvisionAugMix is None:
        raise RuntimeError(
            "torchvision.transforms.AugMix is not available in this environment. "
            "Install a newer torchvision build or disable --use-augmix."
        )

    return transforms.Compose([
        TorchvisionAugMix(
            severity=severity,
            mixture_width=mixture_width,
            chain_depth=chain_depth,
            alpha=alpha,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def load_cifar10_loader(data_root: str,
                        batch_size: int = 128,
                        num_workers: int = 2,
                        use_augmix: bool = False,
                        augmix_severity: int = 3,
                        augmix_mixture_width: int = 3,
                        augmix_chain_depth: int = -1,
                        augmix_alpha: float = 1.0) -> DataLoader:
    transform = build_augmix_transform(
        severity=augmix_severity,
        mixture_width=augmix_mixture_width,
        chain_depth=augmix_chain_depth,
        alpha=augmix_alpha,
    ) if use_augmix else build_clean_transform()

    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_cifar10c_loader(corruption: str,
                         severity: int,
                         data_dir: str = "./data/CIFAR-10-C",
                         batch_size: int = 128) -> DataLoader:
    assert os.path.exists(data_dir), f"CIFAR-10-C not found at {data_dir}"
    images_path = os.path.join(data_dir, f"{corruption}.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    assert os.path.exists(images_path), f"Corruption file not found: {images_path}"

    images = np.load(images_path)
    labels = np.load(labels_path)

    start = (severity - 1) * 10000
    end = severity * 10000
    images = images[start:end]
    labels = labels[start:end]

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    images = images.astype(np.float32) / 255.0
    images = ((images - mean) / std).astype(np.float32)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    labels = torch.from_numpy(labels.astype(np.int64))

    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def load_cifar10p_loaders(perturbation: str,
                          data_dir: str = "./data/CIFAR-10-P",
                          batch_size: int = 128) -> tuple[list[DataLoader], int]:
    assert os.path.exists(data_dir), f"CIFAR-10-P not found at {data_dir}"
    images_path = os.path.join(data_dir, f"{perturbation}.npy")
    assert os.path.exists(images_path), f"Perturbation file not found: {images_path}"

    images = np.load(images_path)
    n_images = 10000

    if images.ndim == 4:
        assert images.shape[0] % n_images == 0, f"Invalid CIFAR-10-P shape: {images.shape}"
        n_steps = images.shape[0] // n_images
        images = images.reshape(n_steps, n_images, 32, 32, 3)
    elif images.ndim == 5:
        if images.shape[1] == n_images:
            n_steps = images.shape[0]
        elif images.shape[0] == n_images:
            n_steps = images.shape[1]
            images = np.transpose(images, (1, 0, 2, 3, 4))
        else:
            raise ValueError(f"Unexpected CIFAR-10-P shape: {images.shape}")
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    labels = None
    for label_fname in ["labels.npy", "cifar10_labels.npy", "labels_hard.npy", "label.npy"]:
        label_path = os.path.join(data_dir, label_fname)
        if os.path.exists(label_path):
            labels = np.load(label_path)
            break

    if labels is None:
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
        labels = np.array(test_set.targets)

    labels = labels[:n_images].astype(np.int64)
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    loaders: list[DataLoader] = []
    for step in range(n_steps):
        step_images = images[step].astype(np.float32) / 255.0
        step_images = ((step_images - mean) / std).astype(np.float32)
        step_images = torch.from_numpy(step_images).permute(0, 3, 1, 2).contiguous()
        step_labels = torch.from_numpy(labels)

        loaders.append(DataLoader(
            TensorDataset(step_images, step_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        ))

    return loaders, n_steps


def evaluate_classifier(model: torch.nn.Module,
                        loader: DataLoader,
                        device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def evaluate_cascade(model_s: torch.nn.Module,
                     model_l: torch.nn.Module,
                     loader: DataLoader,
                     device: torch.device,
                     tau: float = 0.7) -> dict[str, float]:
    model_s.eval()
    model_l.eval()

    total = 0
    correct_s = 0
    correct_l = 0
    correct_cascade = 0
    deferred = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits_s = model_s(images)
            probs_s = F.softmax(logits_s, dim=-1)
            conf_s, pred_s = probs_s.max(dim=-1)

            logits_l = model_l(images)
            pred_l = logits_l.argmax(dim=-1)

            defer_mask = conf_s < tau
            pred_final = pred_s.clone()
            pred_final[defer_mask] = pred_l[defer_mask]

            total += labels.size(0)
            correct_s += (pred_s == labels).sum().item()
            correct_l += (pred_l == labels).sum().item()
            correct_cascade += (pred_final == labels).sum().item()
            deferred += defer_mask.sum().item()

    return {
        "acc_s": correct_s / total if total else 0.0,
        "acc_l": correct_l / total if total else 0.0,
        "cascade_acc": correct_cascade / total if total else 0.0,
        "defer_rate": deferred / total if total else 0.0,
    }


def evaluate_cifar10p(model_s: torch.nn.Module,
                      model_l: torch.nn.Module,
                      loaders: list[DataLoader],
                      device: torch.device,
                      tau: float = 0.7) -> dict[str, float]:
    acc_s_steps = []
    acc_l_steps = []
    cascade_steps = []
    defer_steps = []

    for loader in loaders:
        step_metrics = evaluate_cascade(model_s, model_l, loader, device, tau=tau)
        acc_s_steps.append(step_metrics["acc_s"])
        acc_l_steps.append(step_metrics["acc_l"])
        cascade_steps.append(step_metrics["cascade_acc"])
        defer_steps.append(step_metrics["defer_rate"])

    return {
        "mean_acc_s": float(np.mean(acc_s_steps)),
        "mean_acc_l": float(np.mean(acc_l_steps)),
        "mean_cascade_acc": float(np.mean(cascade_steps)),
        "mean_defer_rate": float(np.mean(defer_steps)),
        "steps": len(loaders),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test CIFAR-10 models with AugMix and optional robustness benchmarks."
    )
    parser.add_argument("--split", choices=["clean", "cifar10c", "cifar10p"], default="clean")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--perturbation", type=str, default="rotate")
    parser.add_argument("--corruption", type=str, default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--use-augmix", action="store_true", help="Apply AugMix on clean CIFAR-10 test images.")
    parser.add_argument("--augmix-severity", type=int, default=3)
    parser.add_argument("--augmix-mixture-width", type=int, default=3)
    parser.add_argument("--augmix-chain-depth", type=int, default=-1)
    parser.add_argument("--augmix-alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--small-checkpoint", type=str, default="model_s_pretrained.pth")
    parser.add_argument("--large-checkpoint", type=str, default="model_l_pretrained.pth")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "Robustness_Cascade"))
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--codecarbon", action="store_true", help="Enable CodeCarbon emissions tracking.")
    parser.add_argument("--codecarbon-output-dir", type=str, default="codecarbon")
    return parser


def _load_models(num_classes: int,
                 small_checkpoint: str,
                 large_checkpoint: str,
                 device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    model_s = SmallCNN(num_classes=num_classes).to(device)
    model_l = get_resnet18(num_classes=num_classes).to(device)

    model_s.load_state_dict(torch.load(small_checkpoint, map_location=device))
    model_l.load_state_dict(torch.load(large_checkpoint, map_location=device))
    model_s.eval()
    model_l.eval()
    return model_s, model_l


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    device = get_device()
    data_dir = args.data_dir or (
        "./data/CIFAR-10-C" if args.split == "cifar10c"
        else "./data/CIFAR-10-P" if args.split == "cifar10p"
        else args.data_root
    )

    wandb_run = None
    if not args.no_wandb:
        if wandb is None:
            print("[WARN] wandb is not installed; continuing without experiment logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                mode=args.wandb_mode,
                config={
                    "split": args.split,
                    "data_root": args.data_root,
                    "data_dir": data_dir,
                    "perturbation": args.perturbation,
                    "corruption": args.corruption,
                    "severity": args.severity,
                    "use_augmix": args.use_augmix,
                    "augmix_severity": args.augmix_severity,
                    "tau": args.tau,
                    "batch_size": args.batch_size,
                },
            )

    tracker = _start_codecarbon_tracker(
        enabled=args.codecarbon,
        output_dir=args.codecarbon_output_dir,
        project_name=f"cifar10_augmix_{args.split}",
    )

    try:
        model_s, model_l = _load_models(
            num_classes=args.num_classes,
            small_checkpoint=args.small_checkpoint,
            large_checkpoint=args.large_checkpoint,
            device=device,
        )

        if args.split == "clean":
            loader = load_cifar10_loader(
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_augmix=args.use_augmix,
                augmix_severity=args.augmix_severity,
                augmix_mixture_width=args.augmix_mixture_width,
                augmix_chain_depth=args.augmix_chain_depth,
                augmix_alpha=args.augmix_alpha,
            )
            cascade_metrics = evaluate_cascade(model_s, model_l, loader, device, tau=args.tau)
            clean_acc_s = evaluate_classifier(model_s, loader, device)
            clean_acc_l = evaluate_classifier(model_l, loader, device)

            results = {
                "split": "clean",
                "use_augmix": args.use_augmix,
                "acc_s": clean_acc_s,
                "acc_l": clean_acc_l,
                **cascade_metrics,
            }

        elif args.split == "cifar10c":
            loader = load_cifar10c_loader(
                corruption=args.corruption,
                severity=args.severity,
                data_dir=data_dir,
                batch_size=args.batch_size,
            )
            cascade_metrics = evaluate_cascade(model_s, model_l, loader, device, tau=args.tau)
            results = {
                "split": "cifar10c",
                "corruption": args.corruption,
                "severity": args.severity,
                **cascade_metrics,
            }

        else:
            loaders, n_steps = load_cifar10p_loaders(
                perturbation=args.perturbation,
                data_dir=data_dir,
                batch_size=args.batch_size,
            )
            results = {
                "split": "cifar10p",
                "perturbation": args.perturbation,
                **evaluate_cifar10p(model_s, model_l, loaders, device, tau=args.tau),
                "n_steps": n_steps,
            }

        output_name = f"augmix_{args.split}_results.pkl"
        with open(output_name, "wb") as f:
            pickle.dump(results, f)

        print(f"\n[INFO] Saved → {output_name}")
        print("[INFO] Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        if wandb_run is not None:
            wandb_run.summary.update(results)
            wandb_run.save(output_name)
            wandb_run.log({"results/file": output_name})
    finally:
        _stop_codecarbon_tracker(tracker, wandb_run, f"augmix_{args.split}")
        if wandb_run is not None:
            wandb_run.finish()
