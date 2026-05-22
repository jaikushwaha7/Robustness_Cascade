import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models_cifar100 import SmallCNN, get_resnet18

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


# ─────────────────────────────────────────────
# 1. COLLECT CONFIDENCE SCORES
# ─────────────────────────────────────────────

def collect_confidence_scores(model_s, loader, device="cpu"):
    """
    Run MS on the test set and collect:
      - confidence scores for correctly predicted samples
      - confidence scores for incorrectly predicted samples

    Returns:
        correct_confs   : np.array of max-softmax scores where MS was right
        incorrect_confs : np.array of max-softmax scores where MS was wrong
    """
    model_s.eval()
    correct_confs, incorrect_confs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model_s(images)
            probs  = F.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

            mask = (preds == labels)
            correct_confs.extend(conf[mask].cpu().numpy())
            incorrect_confs.extend(conf[~mask].cpu().numpy())

    return np.array(correct_confs), np.array(incorrect_confs)


# ─────────────────────────────────────────────
# 2. METRIC: Distributional Overlap (s_o)
#    Equation 9 in the paper — lower is better
# ─────────────────────────────────────────────

def compute_distributional_overlap(correct_confs, incorrect_confs,
                                   n_grid=1000):
    """
    Estimates KDEs over correct/incorrect confidence scores
    and computes the overlap area s_o.

    s_o = integral of min(kde_correct(c), kde_incorrect(c)) dc
    Lower s_o = better separation = better deferral.
    """
    from sklearn.neighbors import KernelDensity

    grid = np.linspace(0, 1, n_grid).reshape(-1, 1)

    kde_c = KernelDensity(bandwidth=0.05, kernel='gaussian')
    kde_c.fit(correct_confs.reshape(-1, 1))

    kde_i = KernelDensity(bandwidth=0.05, kernel='gaussian')
    kde_i.fit(incorrect_confs.reshape(-1, 1))

    dens_c = np.exp(kde_c.score_samples(grid))
    dens_i = np.exp(kde_i.score_samples(grid))

    # Normalize so each integrates to 1
    dens_c /= dens_c.sum() * (1 / n_grid)
    dens_i /= dens_i.sum() * (1 / n_grid)

    s_o = np.minimum(dens_c, dens_i).sum() * (1 / n_grid)
    return s_o, dens_c, dens_i, grid.flatten()


# ─────────────────────────────────────────────
# 3. METRIC: Deferral Performance (s_d)
#    Equation 10 in the paper — higher is better
# ─────────────────────────────────────────────

def compute_deferral_performance(model_s, model_l, loader,
                                  tau_values=None, device="cpu"):
    """
    Sweeps deferral thresholds tau to build the realized deferral curve.
    Computes s_d = normalized area between realized and random deferral.

    Returns:
        deferral_ratios  : array of deferral ratios (x-axis)
        joint_accs_real  : realized joint accuracy at each ratio (black curve)
        joint_accs_rand  : random deferral baseline (red dashed curve)
        joint_accs_ideal : ideal deferral upper bound (green curve)
        s_d              : scalar score (higher is better)
        acc_s            : MS standalone accuracy (yellow dot)
        acc_l            : ML standalone accuracy (purple dot)
    """
    if tau_values is None:
        tau_values = np.linspace(0.0, 1.0, 100)

    model_s.eval()
    model_l.eval()

    # ── Collect all predictions and confidences in one pass ──
    all_confs, all_preds_s, all_preds_l, all_labels = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits_s = model_s(images)
            logits_l = model_l(images)

            probs_s       = F.softmax(logits_s, dim=-1)
            conf_s, pred_s = probs_s.max(dim=-1)
            pred_l         = logits_l.argmax(dim=-1)

            all_confs.append(conf_s.cpu())
            all_preds_s.append(pred_s.cpu())
            all_preds_l.append(pred_l.cpu())
            all_labels.append(labels)

    all_confs   = torch.cat(all_confs).numpy()
    all_preds_s = torch.cat(all_preds_s).numpy()
    all_preds_l = torch.cat(all_preds_l).numpy()
    all_labels  = torch.cat(all_labels).numpy()

    n = len(all_labels)

    # Standalone accuracies
    acc_s = (all_preds_s == all_labels).mean()
    acc_l = (all_preds_l == all_labels).mean()

    # ── Build realized deferral curve by sweeping tau ──
    deferral_ratios, joint_accs_real = [], []

    for tau in tau_values:
        defer_mask = all_confs < tau          # defer when not confident

        # Joint prediction: MS handles confident, ML handles deferred
        joint_preds = all_preds_s.copy()
        joint_preds[defer_mask] = all_preds_l[defer_mask]

        joint_acc    = (joint_preds == all_labels).mean()
        deferral_rate = defer_mask.mean()

        deferral_ratios.append(deferral_rate)
        joint_accs_real.append(joint_acc)

    deferral_ratios  = np.array(deferral_ratios)
    joint_accs_real  = np.array(joint_accs_real)

    # Sort by deferral ratio for clean curves
    sort_idx         = np.argsort(deferral_ratios)
    deferral_ratios  = deferral_ratios[sort_idx]
    joint_accs_real  = joint_accs_real[sort_idx]

    # ── Random deferral baseline (linear interpolation) ──
    joint_accs_rand = acc_s + (acc_l - acc_s) * deferral_ratios

    # ── Ideal deferral curve (Appendix B.3, Equation 11) ──
    # Guard against division by zero when acc_s is close to 1
    error_rate_s = max(1 - acc_s, 1e-6)
    slope        = (acc_l - acc_s) / error_rate_s
    joint_accs_ideal = np.where(
        deferral_ratios <= error_rate_s,
        acc_s + slope * deferral_ratios,
        acc_l
    )
    # Clip to [acc_s, acc_l] to prevent any numerical overshoot
    joint_accs_ideal = np.clip(joint_accs_ideal, acc_s, acc_l)

    # ── s_d: normalized area between realized and random ──
    # Interpolate onto a uniform grid for stable integration
    # (tau sweep produces unevenly spaced deferral ratios)
    uniform_grid = np.linspace(0, 1, 500)
    real_interp  = np.interp(uniform_grid, deferral_ratios, joint_accs_real)
    rand_interp  = np.interp(uniform_grid, deferral_ratios, joint_accs_rand)
    ideal_interp = np.interp(uniform_grid, deferral_ratios, joint_accs_ideal)

    area_real  = np.trapezoid(np.maximum(real_interp  - rand_interp,  0), uniform_grid)
    area_ideal = np.trapezoid(np.maximum(ideal_interp - rand_interp,  0), uniform_grid)
    s_d = float(np.clip(area_real / area_ideal, 0.0, 1.0)) if area_ideal > 1e-8 else 0.0

    return (deferral_ratios, joint_accs_real, joint_accs_rand,
            joint_accs_ideal, s_d, acc_s, acc_l)


# ─────────────────────────────────────────────
# 4. FULL EVALUATION OVER ALL ALPHA VALUES
# ─────────────────────────────────────────────

def evaluate_all_alphas(model_l, val_loader,
                         alphas, num_classes=100, device="cpu",
                         model_dir="models",
                         wandb_run=None):
    """
    Loads each fine-tuned MS checkpoint and computes s_o and s_d.

    Returns:
        results : dict keyed by alpha, each containing all metrics
    """
    results = {}

    metrics_table = None
    if wandb_run is not None:
        metrics_table = wandb.Table(
            columns=["key", "s_o", "s_d", "acc_s", "acc_l"]
        )

    # ── Baseline: pre-trained MS without Gatekeeper fine-tuning ──
    print("Evaluating baseline (no Gatekeeper)...")
    model_s_base = SmallCNN(num_classes=num_classes).to(device)
    model_s_base.load_state_dict(
        torch.load(Path(model_dir) / "model_s_cifar100_pretrained.pth", map_location=device))

    correct_confs, incorrect_confs = collect_confidence_scores(
        model_s_base, val_loader, device)
    s_o, dens_c, dens_i, grid = compute_distributional_overlap(
        correct_confs, incorrect_confs)
    deferral_ratios, accs_real, accs_rand, accs_ideal, s_d, acc_s, acc_l = \
        compute_deferral_performance(
            model_s_base, model_l, val_loader, device=device)

    results["baseline"] = {
        "s_o": s_o, "s_d": s_d, "acc_s": acc_s, "acc_l": acc_l,
        "correct_confs": correct_confs, "incorrect_confs": incorrect_confs,
        "dens_c": dens_c, "dens_i": dens_i, "grid": grid,
        "deferral_ratios": deferral_ratios,
        "accs_real": accs_real, "accs_rand": accs_rand,
        "accs_ideal": accs_ideal,
    }
    print(f"  Baseline → s_o: {s_o:.4f} | s_d: {s_d:.4f} | acc_s: {acc_s:.4f}")
    if wandb_run is not None:
        wandb_run.log({
            "eval/baseline_s_o": s_o,
            "eval/baseline_s_d": s_d,
            "eval/baseline_acc_s": acc_s,
            "eval/baseline_acc_l": acc_l,
        })
        metrics_table.add_data("baseline", s_o, s_d, acc_s, acc_l)

    # ── Gatekeeper fine-tuned MS for each alpha ──
    for alpha in alphas:
        print(f"Evaluating alpha={alpha}...")
        model_s = SmallCNN(num_classes=num_classes).to(device)
        model_s.load_state_dict(
            torch.load(Path(model_dir) / f"model_s_cifar100_gk_alpha{alpha}.pth", map_location=device))

        correct_confs, incorrect_confs = collect_confidence_scores(
            model_s, val_loader, device)
        s_o, dens_c, dens_i, grid = compute_distributional_overlap(
            correct_confs, incorrect_confs)
        deferral_ratios, accs_real, accs_rand, accs_ideal, s_d, acc_s, acc_l = \
            compute_deferral_performance(
                model_s, model_l, val_loader, device=device)

        results[alpha] = {
            "s_o": s_o, "s_d": s_d, "acc_s": acc_s, "acc_l": acc_l,
            "correct_confs": correct_confs, "incorrect_confs": incorrect_confs,
            "dens_c": dens_c, "dens_i": dens_i, "grid": grid,
            "deferral_ratios": deferral_ratios,
            "accs_real": accs_real, "accs_rand": accs_rand,
            "accs_ideal": accs_ideal,
        }
        print(f"  alpha={alpha} → s_o: {s_o:.4f} | s_d: {s_d:.4f} | acc_s: {acc_s:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                f"eval/alpha_{alpha}/s_o": s_o,
                f"eval/alpha_{alpha}/s_d": s_d,
                f"eval/alpha_{alpha}/acc_s": acc_s,
                f"eval/alpha_{alpha}/acc_l": acc_l,
            })
            metrics_table.add_data(str(alpha), s_o, s_d, acc_s, acc_l)

    if wandb_run is not None and metrics_table is not None:
        wandb_run.log({"eval/metrics_table": metrics_table})

    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the CIFAR-100 cascade and optionally log results to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="Robustness_Cascade")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb_run = None
    if not args.no_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "device": device,
                "alphas": [0.9, 0.7, 0.5, 0.3, 0.1],
                "evaluation": "cascade_robustness_cifar100",
            },
        )
    elif not args.no_wandb:
        print("Weights & Biases is not installed, continuing without experiment logging.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(test_set, batch_size=128,
                            shuffle=False, num_workers=2)

    # Load ML (fixed across all evaluations)
    model_l = get_resnet18(num_classes=100).to(device)
    model_l.load_state_dict(
        torch.load(Path("models") / "model_l_cifar100_pretrained.pth", map_location=device))
    model_l.eval()

    alphas  = [0.9, 0.7, 0.5, 0.3, 0.1]
    results = evaluate_all_alphas(model_l, val_loader,
                                   alphas, num_classes=100, device=device,
                                   model_dir="models",
                                   wandb_run=wandb_run)

    # Save results for plotting
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "eval_results_cifar100.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    if wandb_run is not None:
        artifact = wandb.Artifact("eval_results_cifar100", type="evaluation")
        artifact.add_file(results_path)
        wandb_run.log_artifact(artifact)
        wandb_run.summary["eval/result_file"] = results_path

    print(f"\nSaved → {results_path}")
    print("\nSummary:")
    print(f"{'Key':<12} {'s_o':>8} {'s_d':>8} {'acc_s':>8}")
    print("-" * 40)
    for key in ["baseline"] + alphas:
        r = results[key]
        print(f"{str(key):<12} {r['s_o']:>8.4f} {r['s_d']:>8.4f} {r['acc_s']:>8.4f}")

    if wandb_run is not None:
        wandb_run.finish()
