import argparse
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models_cifar100 import SmallCNN, get_resnet18
from gatekeeper_loss import GatekeeperLoss

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover - optional dependency
    EmissionsTracker = None

# ============================================================
# CONFIG — CIFAR-100
# ============================================================
 
NUM_CLASSES = 100
BATCH_SIZE = 128
 
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
 
 
# ============================================================
# DATA
# ============================================================
 
def _ensure_cifar100_data_exists(data_root: str) -> Path:
    cifar100_root = Path(data_root) / "cifar-100-python"
    if not cifar100_root.exists():
        raise FileNotFoundError(
            f"CIFAR-100 data not found at {cifar100_root}. "
            "Place the extracted CIFAR-100 dataset there before running, "
            "or add a download step if network access is available."
        )

    return cifar100_root


def get_loaders(data_root: str):
    _ensure_cifar100_data_exists(data_root)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
 
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
 
    train_set = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
 
    val_set = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform_val,
    )
 
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
 
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
 
    return train_loader, val_loader
# ─────────────────────────────────────────────
# STAGE 1a — Train MS (SmallCNN)
# Adam + StepLR — works well for small CNNs
# ─────────────────────────────────────────────

def _log_wandb(wandb_run: Any, data: dict[str, float], step: Optional[int] = None) -> None:
    if wandb_run is not None:
        wandb_run.log(data, step=step)


def _start_codecarbon_tracker(enabled: bool,
                              output_dir: str,
                              project_name: str) -> Any:
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


def _stop_codecarbon_tracker(tracker: Any,
                             wandb_run: Any,
                             metric_prefix: str,
                             step: Optional[int] = None) -> Optional[float]:
    if tracker is None:
        return None

    emissions_kg = tracker.stop()
    print(f"[{metric_prefix}] CodeCarbon emissions: {emissions_kg:.6f} kg CO2e")
    if wandb_run is not None:
        wandb_run.log({f"{metric_prefix}/emissions_kg": emissions_kg}, step=step)
    return emissions_kg


def train_small_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int = 50,
                      lr: float = 1e-3,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu",
                      wandb_run: Any = None) -> nn.Module:
    """
    Trains SmallCNN (MS) with Adam + StepLR.
    Target: ~72-75% on CIFAR-100.

    Args:
        model        : SmallCNN
        train_loader : augmented training DataLoader
        val_loader   : clean validation DataLoader
        epochs       : 50 epochs enough for SmallCNN to converge
        lr           : 1e-3 works well with Adam for small CNNs
        device       : 'cpu' or 'cuda'
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Step down lr at epoch 25 and 40
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 40], gamma=0.1)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        _log_wandb(wandb_run, {
            "ms/train_loss": train_loss / len(train_loader),
            "ms/train_acc": train_correct / train_total,
            "ms/val_acc": val_acc,
            "ms/lr": scheduler.get_last_lr()[0],
            "epoch": epoch + 1,
        }, step=epoch + 1)
        print(f"[MS] Epoch [{epoch+1:02d}/{epochs}]  "
              f"Loss: {train_loss/len(train_loader):.4f}  "
              f"Train: {train_correct/train_total:.4f}  "
              f"Val: {val_acc:.4f}")

    return model


# ─────────────────────────────────────────────
# STAGE 1b — Train ML (ResNet-18)
# SGD + CosineAnnealingLR — standard recipe
# Target: 93%+ on CIFAR-100
# ─────────────────────────────────────────────

def train_large_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      epochs: int = 100,
                      lr: float = 0.1,
                      device: str ="cuda" if torch.cuda.is_available() else "cpu",
                      wandb_run: Any = None) -> nn.Module:
    """
    Trains ResNet-18 (ML) with SGD + momentum + cosine schedule.
    This is the standard recipe that gets ResNet-18 to 93%+ on CIFAR-100.

    Args:
        model        : ResNet-18
        train_loader : augmented training DataLoader
        val_loader   : clean validation DataLoader
        epochs       : 100 epochs for full convergence
        lr           : 0.1 starting lr for SGD (standard for ResNets)
        device       : 'cpu' or 'cuda'
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # Print every 10 epochs to avoid too much output
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_acc = evaluate(model, val_loader, device)
            _log_wandb(wandb_run, {
                "ml/train_loss": train_loss / len(train_loader),
                "ml/train_acc": train_correct / train_total,
                "ml/val_acc": val_acc,
                "ml/lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }, step=epoch + 1)
            print(f"[ML] Epoch [{epoch+1:03d}/{epochs}]  "
                  f"Loss: {train_loss/len(train_loader):.4f}  "
                  f"Train: {train_correct/train_total:.4f}  "
                  f"Val: {val_acc:.4f}")
        else:
            _log_wandb(wandb_run, {
                "ml/train_loss": train_loss / len(train_loader),
                "ml/train_acc": train_correct / train_total,
                "ml/lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }, step=epoch + 1)

    return model


# ─────────────────────────────────────────────
# STAGE 2 — Gatekeeper Fine-tuning (MS only)
# ML is frozen — only MS gets updated
# ─────────────────────────────────────────────

def finetune_gatekeeper(model_s: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         alpha: float = 0.5,
                        num_classes: int = 100,
                         epochs: int = 30,
                         lr: float = 3e-4,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu",
                         wandb_run: Any = None) -> nn.Module:
    """
    Fine-tune MS with GatekeeperLoss.
    MS should already be pre-trained (Stage 1) before calling this.
    ML is never touched here.

    Args:
        model_s      : pre-trained SmallCNN (MS)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        alpha        : Gatekeeper trade-off in (0, 1)
                       paper sweeps: [0.1, 0.3, 0.5, 0.7, 0.9]
        num_classes  : number of output classes
        epochs       : 30 epochs — enough for confidence to reshape
        lr           : 3e-4 — slightly higher than before for faster reshaping
        device       : 'cpu' or 'cuda'
    """
    model_s = model_s.to(device)
    criterion = GatekeeperLoss(alpha=alpha, num_classes=num_classes)
    optimizer = optim.Adam(model_s.parameters(), lr=lr, weight_decay=1e-4)
    # Gentle decay in the second half of fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    print(f"\n── Gatekeeper Fine-tuning | alpha={alpha} ──")

    for epoch in range(epochs):
        model_s.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model_s(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        val_acc = evaluate(model_s, val_loader, device)
        _log_wandb(wandb_run, {
            "gk/alpha": alpha,
            "gk/train_loss": total_loss / len(train_loader),
            "gk/val_acc": val_acc,
            "gk/lr": scheduler.get_last_lr()[0],
            "epoch": epoch + 1,
        }, step=epoch + 1)
        print(f"  Epoch [{epoch+1:02d}/{epochs}]  "
              f"GK Loss: {total_loss/len(train_loader):.4f}  "
              f"Val Acc (MS): {val_acc:.4f}")

    return model_s


# ─────────────────────────────────────────────
# HELPER — Accuracy Evaluation
# ─────────────────────────────────────────────

def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cpu") -> float:
    """Returns accuracy of model on the given loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def _parse_alpha_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CIFAR-100 cascade with optional Weights & Biases logging.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--small-epochs", type=int, default=50)
    parser.add_argument("--large-epochs", type=int, default=100)
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--alpha-values", type=str, default="0.9,0.7,0.5,0.3,0.1")
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "Robustness_Cascade"))
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", type=str, default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--codecarbon-output-dir", type=str, default="codecarbon")
    parser.add_argument("--codecarbon-project-name", type=str, default="Robustness_Cascade")
    parser.add_argument("--no-codecarbon", action="store_true", help="Disable CodeCarbon emissions tracking.")
    return parser


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    args = _build_arg_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Transforms ──
    # Augmented for training — helps both MS and ML generalise better
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    # Clean for validation — no augmentation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    _ensure_cifar100_data_exists(args.data_root)

    train_set = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(test_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    wandb_run = None
    if not args.no_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "device": device,
                "data_root": args.data_root,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "small_epochs": args.small_epochs,
                "large_epochs": args.large_epochs,
                "fine_tune_epochs": args.fine_tune_epochs,
                "alpha_values": _parse_alpha_values(args.alpha_values),
                "dataset": "CIFAR-100",
                "model": "Robustness Cascade CIFAR-100",
            },
        )
    elif not args.no_wandb:
        print("Weights & Biases is not installed, continuing without experiment logging.")

    codecarbon_enabled = (not args.no_codecarbon) and (EmissionsTracker is not None)
    if not args.no_codecarbon and EmissionsTracker is None:
        print("CodeCarbon is not installed, continuing without emissions tracking.")

    total_emissions_kg = 0.0
    os.makedirs("models", exist_ok=True)

    try:
        # ── Stage 1a: Train MS (SmallCNN) ──
        # 
        print("\n=== Stage 1a: Training MS (SmallCNN) on CIFAR-100 ===")
        ms_tracker = _start_codecarbon_tracker(
            codecarbon_enabled,
            args.codecarbon_output_dir,
            f"{args.codecarbon_project_name}-cifar100-ms",
        )
        model_s = SmallCNN(num_classes=100)
        model_s = train_small_model(model_s, train_loader, val_loader,
                                    epochs=args.small_epochs, lr=1e-3,
                                    device=device, wandb_run=wandb_run)
        ms_emissions = _stop_codecarbon_tracker(ms_tracker, wandb_run, "ms", step=args.small_epochs)
        if ms_emissions is not None:
            total_emissions_kg += ms_emissions
        model_s_path = os.path.join("models", "model_s_cifar100_pretrained.pth")
        torch.save(model_s.state_dict(), model_s_path)
        print(f"Saved → {model_s_path}")
        if wandb_run is not None:
            wandb_run.save(model_s_path)

        # ── Stage 1b: Train ML (ResNet-18) ──
        # 
        print("\n=== Stage 1b: Training ML (ResNet-18) on CIFAR-100 ===")
        ml_tracker = _start_codecarbon_tracker(
            codecarbon_enabled,
            args.codecarbon_output_dir,
            f"{args.codecarbon_project_name}-cifar100-ml",
        )
        model_l = get_resnet18(num_classes=100)
        model_l = train_large_model(model_l, train_loader, val_loader,
                                    epochs=args.large_epochs, lr=0.1,
                                    device=device, wandb_run=wandb_run)
        ml_emissions = _stop_codecarbon_tracker(ml_tracker, wandb_run, "ml", step=args.large_epochs)
        if ml_emissions is not None:
            total_emissions_kg += ml_emissions
        model_l_path = os.path.join("models", "model_l_cifar100_pretrained.pth")
        torch.save(model_l.state_dict(), model_l_path)
        print(f"Saved → {model_l_path}")
        if wandb_run is not None:
            wandb_run.save(model_l_path)

        # ── Stage 2: Fine-tune MS with Gatekeeper loss ──
        print("\n=== Stage 2: Gatekeeper Fine-tuning on CIFAR-100 ===")
        for alpha in _parse_alpha_values(args.alpha_values):
            # Always reload fresh Stage 1 MS — each alpha starts from same baseline
            gk_tracker = _start_codecarbon_tracker(
                codecarbon_enabled,
                args.codecarbon_output_dir,
                f"{args.codecarbon_project_name}-cifar100-gk-alpha{alpha}",
            )
            model_s_ft = SmallCNN(num_classes=100)
            model_s_ft.load_state_dict(
                torch.load(model_s_path, map_location=device))

            model_s_ft = finetune_gatekeeper(
                model_s_ft, train_loader, val_loader,
                alpha=alpha, num_classes=100,
                epochs=args.fine_tune_epochs, lr=3e-4,
                device=device, wandb_run=wandb_run
            )
            checkpoint_name = os.path.join("models", f"model_s_cifar100_gk_alpha{alpha}.pth")
            torch.save(model_s_ft.state_dict(), checkpoint_name)
            print(f"Saved → {checkpoint_name}")
            if wandb_run is not None:
                wandb_run.save(checkpoint_name)
            gk_emissions = _stop_codecarbon_tracker(gk_tracker, wandb_run, f"gk/alpha_{alpha}", step=args.fine_tune_epochs)
            if gk_emissions is not None:
                total_emissions_kg += gk_emissions

        if wandb_run is not None and codecarbon_enabled:
            wandb_run.summary["codecarbon/total_emissions_kg"] = total_emissions_kg
    finally:
        if wandb_run is not None:
            wandb_run.finish()
