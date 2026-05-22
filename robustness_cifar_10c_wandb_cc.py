import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import SmallCNN, get_resnet18
from evaluate import (collect_confidence_scores,
                      compute_distributional_overlap,
                      compute_deferral_performance)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover - optional dependency
    EmissionsTracker = None


# ─────────────────────────────────────────────
# ALL 19 CIFAR-10-C CORRUPTIONS
# ─────────────────────────────────────────────

CORRUPTIONS = [
    # Noise
    "gaussian_noise", "shot_noise", "impulse_noise",
    # Blur
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    # Weather
    "snow", "frost", "fog", "brightness",
    # Digital
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    # Extra
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]

SEVERITIES = [1, 2, 3, 4, 5]


def _parse_alpha_values(alpha_values: str) -> list[float]:
    return [float(value.strip()) for value in alpha_values.split(",") if value.strip()]


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


def _alpha_label(key: Any) -> str:
    return "baseline" if key == "baseline" else f"alpha={key}"


def _log_robustness_summary(wandb_run: Any,
                            rob_results: dict[Any, dict[str, dict[int, dict[str, float]]]],
                            alphas: list[float]) -> None:
    if wandb_run is None:
        return

    table = wandb.Table(columns=[
        "setting", "corruption", "severity",
        "s_o", "s_d", "acc_s",
        "cascade_acc", "cascade_gain_pct",
    ])
    for key in ["baseline"] + alphas:
        setting = _alpha_label(key)
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                metrics = rob_results[key][corruption][severity]
                table.add_data(
                    setting,
                    corruption,
                    severity,
                    metrics["s_o"],
                    metrics["s_d"],
                    metrics["acc_s"],
                    metrics["cascade_acc"],
                    metrics["cascade_gain_pct"],
                )

    wandb_run.log({"robustness/results_table": table})

    for key in ["baseline"] + alphas:
        setting = _alpha_label(key)
        values = {
            metric: np.mean([
                rob_results[key][corruption][severity][metric]
                for corruption in CORRUPTIONS
                for severity in SEVERITIES
            ])
            for metric in ("s_o", "s_d", "acc_s")
        }
        wandb_run.summary[f"robustness/{setting}/mean_s_o"] = float(values["s_o"])
        wandb_run.summary[f"robustness/{setting}/mean_s_d"] = float(values["s_d"])
        wandb_run.summary[f"robustness/{setting}/mean_acc_s"] = float(values["acc_s"])
        wandb_run.summary[f"robustness/{setting}/mean_cascade_acc"] = float(np.mean([
            rob_results[key][corruption][severity]["cascade_acc"]
            for corruption in CORRUPTIONS
            for severity in SEVERITIES
        ]))
        wandb_run.summary[f"robustness/{setting}/mean_cascade_gain_pct"] = float(np.mean([
            rob_results[key][corruption][severity]["cascade_gain_pct"]
            for corruption in CORRUPTIONS
            for severity in SEVERITIES
        ]))


def _cascade_summary_from_deferral(deferral_ratios: np.ndarray,
                                   accs_real: np.ndarray,
                                   acc_s: float) -> tuple[float, float, float]:
    best_index = int(np.argmax(accs_real))
    cascade_acc = float(accs_real[best_index])
    best_deferral_ratio = float(deferral_ratios[best_index])
    gain_pct = 0.0 if acc_s == 0 else float(((cascade_acc - acc_s) / acc_s) * 100.0)
    return cascade_acc, best_deferral_ratio, gain_pct


def _cascade_tau_curve(model_s, model_l, loader,
                       tau_values: list[float],
                       device: str = "cpu") -> tuple[float, list[dict[str, float]]]:
    model_s.eval()
    model_l.eval()

    all_confs, all_preds_s, all_preds_l, all_labels = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits_s = model_s(images)
            logits_l = model_l(images)

            probs_s = F.softmax(logits_s, dim=-1)
            conf_s, pred_s = probs_s.max(dim=-1)
            pred_l = logits_l.argmax(dim=-1)

            all_confs.append(conf_s.cpu())
            all_preds_s.append(pred_s.cpu())
            all_preds_l.append(pred_l.cpu())
            all_labels.append(labels)

    all_confs = torch.cat(all_confs).numpy()
    all_preds_s = torch.cat(all_preds_s).numpy()
    all_preds_l = torch.cat(all_preds_l).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc_s = float((all_preds_s == all_labels).mean())
    tau_curve: list[dict[str, float]] = []

    for tau in tau_values:
        defer_mask = all_confs < tau
        joint_preds = all_preds_s.copy()
        joint_preds[defer_mask] = all_preds_l[defer_mask]

        cascade_acc = float((joint_preds == all_labels).mean())
        deferral_rate = float(defer_mask.mean())
        gain_pct = 0.0 if acc_s == 0 else float(((cascade_acc - acc_s) / acc_s) * 100.0)

        tau_curve.append({
            "tau": float(tau),
            "cascade_acc": cascade_acc,
            "deferral_rate": deferral_rate,
            "cascade_gain_pct": gain_pct,
        })

    return acc_s, tau_curve


def _plot_stage_emissions(stage_emissions: list[tuple[str, float]],
                          save_path: str = "plot_stage_emissions.png",
                          wandb_run: Any = None) -> None:
    if not stage_emissions:
        return

    stages = [stage for stage, _ in stage_emissions]
    emissions = np.array([value for _, value in stage_emissions], dtype=float)
    pct_changes: list[Optional[float]] = [None]
    for index in range(1, len(emissions)):
        previous = emissions[index - 1]
        current = emissions[index]
        if previous == 0:
            pct_changes.append(None)
        else:
            pct_changes.append(((current - previous) / previous) * 100.0)

    colors = []
    for index, emission in enumerate(emissions):
        if index == 0:
            colors.append("#4C78A8")
        else:
            change = pct_changes[index]
            colors.append("#E15759" if change is not None and change > 0 else "#59A14F")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(stages)), emissions, color=colors, width=0.65)

    ax.set_title("CodeCarbon Emissions by Stage", fontsize=14, fontweight="bold")
    ax.set_ylabel("Emissions (kg CO2e)", fontsize=12)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for index, bar in enumerate(bars):
        height = bar.get_height()
        change = pct_changes[index]
        label = f"{height:.4f} kg"
        if change is not None:
            direction = "increased" if change > 0 else "decreased"
            label = f"{label}\n{direction} {abs(change):.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    if wandb_run is not None:
        wandb_run.log({"robustness/stage_emissions_plot": wandb.Image(save_path)})
        table = wandb.Table(columns=["stage", "emissions_kg", "pct_change_vs_prev"])
        for index, stage in enumerate(stages):
            change = pct_changes[index]
            table.add_data(stage, float(emissions[index]), None if change is None else float(change))
        wandb_run.log({"robustness/stage_emissions_table": table})


def _run_tracked_stage(stage_name: str,
                       enabled: bool,
                       output_dir: str,
                       project_name: str,
                       wandb_run: Any,
                       fn: Any) -> tuple[Any, Optional[float]]:
    tracker = _start_codecarbon_tracker(enabled, output_dir, f"{project_name}-{stage_name}")
    result = None
    try:
        result = fn()
    finally:
        emissions = _stop_codecarbon_tracker(tracker, wandb_run, stage_name)
    return result, emissions


def plot_cascade_accuracy_gain(rob_results, alphas,
                               save_path="plot_cascade_accuracy_gain.png",
                               wandb_run: Any = None):
    """
    Show the best realized cascade accuracy and its percentage gain over the
    standalone small-model accuracy, averaged across corruptions for each severity.
    """
    keys = ["baseline"] + alphas
    labels = ["Baseline"] + [f"α={a}" for a in alphas]
    colors = ["#E8855A", "#C6DBEF", "#9ECAE1", "#6BAED6", "#2171B5", "#08306B"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title, ylabel in zip(
        axes,
        ["cascade_acc", "cascade_gain_pct"],
        ["Peak Cascade Accuracy", "Cascade Accuracy Gain vs $M_S$ (%)"],
        ["Accuracy", "Gain (%)"],
    ):
        for key, label, color in zip(keys, labels, colors):
            mean_per_severity = []
            for sev in SEVERITIES:
                vals = [rob_results[key][c][sev][metric] for c in CORRUPTIONS]
                mean_per_severity.append(np.mean(vals))

            ls = "--" if key == "baseline" else "-"
            ax.plot(SEVERITIES, mean_per_severity,
                    marker='o', label=label,
                    color=color, linestyle=ls, linewidth=2)

        ax.set_xlabel("Corruption Severity", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(SEVERITIES)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cascade Accuracy and Improvement Across Corruption Severity",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    if wandb_run is not None:
        wandb_run.log({"robustness/cascade_accuracy_gain_plot": wandb.Image(save_path)})
    print(f"Saved → {save_path}")


def plot_tau_accuracy_tradeoff(rob_results, alphas, tau_values,
                               save_path="plot_tau_accuracy_tradeoff.png",
                               wandb_run: Any = None):
    """
    Plot mean cascade accuracy and mean gain percentage for each tau value.
    Curves are averaged across corruptions and severities for each alpha.
    """
    keys = ["baseline"] + alphas
    labels = ["Baseline"] + [f"α={a}" for a in alphas]
    colors = ["#E8855A", "#C6DBEF", "#9ECAE1", "#6BAED6", "#2171B5", "#08306B"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = ["cascade_acc", "cascade_gain_pct"]
    titles = ["Cascade Accuracy vs Tau", "Cascade Gain vs Tau (%)"]
    ylabels = ["Accuracy", "Gain (%)"]

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        for key, label, color in zip(keys, labels, colors):
            mean_per_tau = []
            for tau_index, tau in enumerate(tau_values):
                values = [
                    rob_results[key][corruption][severity]["tau_curve"][tau_index][metric]
                    for corruption in CORRUPTIONS
                    for severity in SEVERITIES
                ]
                mean_per_tau.append(float(np.mean(values)))

            ls = "--" if key == "baseline" else "-"
            ax.plot(tau_values, mean_per_tau,
                    marker='o', label=label,
                    color=color, linestyle=ls, linewidth=2)

        ax.set_xlabel("Tau", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Tau Sweep vs Cascade Accuracy and Gain",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    if wandb_run is not None:
        wandb_run.log({"robustness/tau_accuracy_tradeoff": wandb.Image(save_path)})
        table = wandb.Table(columns=["key", "tau", "cascade_acc", "cascade_gain_pct"])
        for key, label in zip(keys, labels):
            for tau_index, tau in enumerate(tau_values):
                acc_values = [
                    rob_results[key][corruption][severity]["tau_curve"][tau_index]["cascade_acc"]
                    for corruption in CORRUPTIONS
                    for severity in SEVERITIES
                ]
                gain_values = [
                    rob_results[key][corruption][severity]["tau_curve"][tau_index]["cascade_gain_pct"]
                    for corruption in CORRUPTIONS
                    for severity in SEVERITIES
                ]
                table.add_data(label, float(tau), float(np.mean(acc_values)), float(np.mean(gain_values)))
        wandb_run.log({"robustness/tau_accuracy_table": table})
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# 1. LOAD CIFAR-10-C
# ─────────────────────────────────────────────

def load_cifar10c(corruption, severity, data_dir="./data/CIFAR-10-C"):
    """
    Loads a specific corruption + severity from CIFAR-10-C.

    CIFAR-10-C structure:
      data_dir/
        gaussian_noise.npy   # shape (50000, 32, 32, 3) — all 5 severities stacked
        shot_noise.npy
        ...
        labels.npy           # shape (50000,) — same labels repeated 5 times

    Each severity has 10000 images:
      severity 1 → indices [0,     10000)
      severity 2 → indices [10000, 20000)
      ...
      severity 5 → indices [40000, 50000)

    Download from: https://zenodo.org/record/2535967
    """
    assert os.path.exists(data_dir), (
        f"CIFAR-10-C not found at {data_dir}.\n"
        f"Download from https://zenodo.org/record/2535967\n"
        f"and extract to {data_dir}"
    )

    images_path = os.path.join(data_dir, f"{corruption}.npy")
    labels_path = os.path.join(data_dir, "labels.npy")

    assert os.path.exists(images_path), \
        f"Corruption file not found: {images_path}"

    images = np.load(images_path)   # (50000, 32, 32, 3) uint8
    labels = np.load(labels_path)   # (50000,)

    # Slice out the correct severity block
    start = (severity - 1) * 10000
    end   =  severity      * 10000
    images = images[start:end]      # (10000, 32, 32, 3)
    labels = labels[start:end]      # (10000,)

    # Normalise to tensor — same stats as CIFAR-10 training
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])

    images = images.astype(np.float32) / 255.0                    # [0, 1]
    images = (images - mean).astype(np.float32) / std             # normalise
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    labels = torch.tensor(labels).long()

    dataset = TensorDataset(images, labels)
    loader  = DataLoader(dataset, batch_size=128,
                         shuffle=False, num_workers=2)
    return loader


# ─────────────────────────────────────────────
# 2. EVALUATE ONE MODEL ON ONE CORRUPTION
# ─────────────────────────────────────────────

def evaluate_corruption(model_s, model_l, corruption,
                         severity, data_dir, device="cpu",
                         tau_values: Optional[list[float]] = None):
    """
    Returns s_o, s_d, acc_s, cascade_acc, cascade_deferral, cascade_gain_pct
    for a given corruption + severity.
    """
    loader = load_cifar10c(corruption, severity, data_dir)

    correct_confs, incorrect_confs = collect_confidence_scores(
        model_s, loader, device)

    s_o, _, _, _ = compute_distributional_overlap(
        correct_confs, incorrect_confs)

    deferral_ratios, accs_real, _, _, s_d, acc_s, _ = compute_deferral_performance(
        model_s, model_l, loader, device=device)

    cascade_acc, cascade_deferral_ratio, cascade_gain_pct = _cascade_summary_from_deferral(
        deferral_ratios, accs_real, acc_s)

    tau_curve = []
    if tau_values:
        _, tau_curve = _cascade_tau_curve(model_s, model_l, loader, tau_values, device=device)

    return s_o, s_d, acc_s, cascade_acc, cascade_deferral_ratio, cascade_gain_pct, tau_curve


# ─────────────────────────────────────────────
# 3. FULL ROBUSTNESS SWEEP
# All corruptions × all severities × all alphas
# ─────────────────────────────────────────────

def run_robustness_evaluation(alphas, data_dir="./data/CIFAR-10-C",
                               num_classes=10, device="cpu",
                               tau_values: Optional[list[float]] = None):
    """
    Runs the full robustness evaluation:
      - For each alpha (+ baseline)
      - For each of 19 corruptions
      - For each of 5 severity levels
      → Records s_o, s_d, acc_s

    Returns:
        rob_results : nested dict
                      rob_results[key][corruption][severity]
                             = {"s_o": ..., "s_d": ..., "acc_s": ...,
                                 "cascade_acc": ..., "cascade_deferral": ...,
                                 "cascade_gain_pct": ..., "tau_curve": [...]} 
    """
    keys  = ["baseline"] + alphas
    ckpts = {"baseline": "model_s_pretrained.pth"}
    ckpts.update({a: f"model_s_gk_alpha{a}.pth" for a in alphas})

    # Load ML once — shared across all evaluations
    model_l = get_resnet18(num_classes=num_classes).to(device)
    model_l.load_state_dict(
        torch.load("model_l_pretrained.pth", map_location=device))
    model_l.eval()

    rob_results = {}

    for key in keys:
        label = "baseline" if key == "baseline" else f"alpha={key}"
        print(f"\n{'='*50}")
        print(f"Evaluating: {label}")
        print(f"{'='*50}")

        model_s = SmallCNN(num_classes=num_classes).to(device)
        model_s.load_state_dict(
            torch.load(ckpts[key], map_location=device))
        model_s.eval()

        rob_results[key] = {}

        for corruption in CORRUPTIONS:
            rob_results[key][corruption] = {}
            for severity in SEVERITIES:
                s_o, s_d, acc_s, cascade_acc, cascade_deferral, cascade_gain_pct, tau_curve = evaluate_corruption(
                    model_s, model_l, corruption,
                    severity, data_dir, device,
                    tau_values=tau_values)

                rob_results[key][corruption][severity] = {
                    "s_o":   s_o,
                    "s_d":   s_d,
                    "acc_s": acc_s,
                    "cascade_acc": cascade_acc,
                    "cascade_deferral": cascade_deferral,
                    "cascade_gain_pct": cascade_gain_pct,
                    "tau_curve": tau_curve,
                }
                print(f"  {corruption:<25} sev={severity}  "
                      f"s_o={s_o:.4f}  s_d={s_d:.4f}  acc_s={acc_s:.4f}  "
                      f"cascade_acc={cascade_acc:.4f}  gain={cascade_gain_pct:+.2f}%")

    return rob_results


# ─────────────────────────────────────────────
# 4. PLOT: Mean Corruption Error (mCE) style
#    acc_s averaged across all corruptions
#    per severity — for each alpha
# ─────────────────────────────────────────────

def plot_robustness_vs_severity(rob_results, alphas,
                                 save_path="plot_robustness_severity.png",
                                 wandb_run: Any = None):
    """
    Three subplots (s_o, s_d, acc_s) × x-axis = severity level.
    Each line = one alpha value.
    Shows how metrics degrade as severity increases.
    """
    keys   = ["baseline"] + alphas
    labels = ["Baseline"] + [f"α={a}" for a in alphas]
    colors = ["#E8855A", "#C6DBEF", "#9ECAE1",
              "#6BAED6", "#2171B5", "#08306B"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics   = ["s_o", "s_d", "acc_s"]
    titles    = ["Distributional Overlap $s_o$ ↓",
                 "Deferral Performance $s_d$ ↑",
                 "Small Model Accuracy $acc(M_S)$ ↑"]

    for ax, metric, title in zip(axes, metrics, titles):
        for key, label, color in zip(keys, labels, colors):
            # Average across all 19 corruptions per severity
            mean_per_severity = []
            for sev in SEVERITIES:
                vals = [rob_results[key][c][sev][metric]
                        for c in CORRUPTIONS]
                mean_per_severity.append(np.mean(vals))

            ls = "--" if key == "baseline" else "-"
            ax.plot(SEVERITIES, mean_per_severity,
                    marker='o', label=label,
                    color=color, linestyle=ls, linewidth=2)

        ax.set_xlabel("Corruption Severity", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(SEVERITIES)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Under Natural Corruptions — CIFAR-10-C",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    if wandb_run is not None:
        wandb_run.log({"robustness/plot_severity": wandb.Image(save_path)})
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# 5. PLOT: Heatmap — acc_s per corruption × severity
#    for a chosen alpha vs baseline
# ─────────────────────────────────────────────

def plot_corruption_heatmap(rob_results, key="baseline",
                             metric="acc_s",
                             save_path="plot_heatmap.png",
                             wandb_run: Any = None):
    """
    Heatmap of a chosen metric across all corruptions (rows)
    and severities (columns) for a given alpha.
    Useful for spotting which corruptions hurt most.
    """
    data = np.zeros((len(CORRUPTIONS), len(SEVERITIES)))
    for i, c in enumerate(CORRUPTIONS):
        for j, s in enumerate(SEVERITIES):
            data[i, j] = rob_results[key][c][s][metric]

    label = "Baseline" if key == "baseline" else f"α={key}"
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(data, aspect='auto',
                   cmap='RdYlGn' if metric != "s_o" else 'RdYlGn_r')

    ax.set_xticks(range(len(SEVERITIES)))
    ax.set_xticklabels([f"Sev {s}" for s in SEVERITIES], fontsize=10)
    ax.set_yticks(range(len(CORRUPTIONS)))
    ax.set_yticklabels(CORRUPTIONS, fontsize=9)
    ax.set_title(f"{metric} — {label}", fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    if wandb_run is not None:
        wandb_run.log({f"robustness/heatmap_{key}_{metric}": wandb.Image(save_path)})
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate robustness on CIFAR-10-C with optional Weights & Biases and CodeCarbon tracking."
    )
    parser.add_argument("--data-dir", type=str, default="./data/CIFAR-10-C")
    parser.add_argument("--alphas", type=str, default="0.9,0.7,0.5,0.3,0.1")
    parser.add_argument("--tau-values", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--heatmap-alpha", type=float, default=0.1)
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "Robustness_Cascade"))
    parser.add_argument("--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", type=str, default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--codecarbon-output-dir", type=str, default="codecarbon")
    parser.add_argument("--codecarbon-project-name", type=str, default="Robustness_Cascade")
    parser.add_argument("--no-codecarbon", action="store_true", help="Disable CodeCarbon emissions tracking.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = _parse_alpha_values(args.alphas)
    tau_values = _parse_alpha_values(args.tau_values)

    wandb_run = None
    if not args.no_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "device": device,
                "data_dir": args.data_dir,
                "alphas": alphas,
                "heatmap_alpha": args.heatmap_alpha,
                "corruptions": CORRUPTIONS,
                "severities": SEVERITIES,
                "model": "Robustness Cascade",
            },
        )
    elif not args.no_wandb:
        print("Weights & Biases is not installed, continuing without experiment logging.")

    codecarbon_enabled = (not args.no_codecarbon) and (EmissionsTracker is not None)
    if not args.no_codecarbon and EmissionsTracker is None:
        print("CodeCarbon is not installed, continuing without emissions tracking.")

    stage_emissions: list[tuple[str, float]] = []

    try:
        rob_results, emissions = _run_tracked_stage(
            "robustness_evaluation",
            codecarbon_enabled,
            args.codecarbon_output_dir,
            args.codecarbon_project_name,
            wandb_run,
            lambda: run_robustness_evaluation(alphas, data_dir=args.data_dir, device=device, tau_values=tau_values),
        )
        if emissions is not None:
            stage_emissions.append(("Robustness evaluation", emissions))

        with open("rob_results.pkl", "wb") as f:
            pickle.dump(rob_results, f)
        print("\nSaved → rob_results.pkl")
        if wandb_run is not None:
            wandb_run.save("rob_results.pkl")

        _log_robustness_summary(wandb_run, rob_results, alphas)

        _, emissions = _run_tracked_stage(
            "severity_plot",
            codecarbon_enabled,
            args.codecarbon_output_dir,
            args.codecarbon_project_name,
            wandb_run,
            lambda: plot_robustness_vs_severity(
                rob_results, alphas,
                save_path="plot_robustness_severity.png",
                wandb_run=wandb_run,
            ),
        )
        if emissions is not None:
            stage_emissions.append(("Severity plot", emissions))

        _, emissions = _run_tracked_stage(
            "baseline_heatmap",
            codecarbon_enabled,
            args.codecarbon_output_dir,
            args.codecarbon_project_name,
            wandb_run,
            lambda: plot_corruption_heatmap(
                rob_results, key="baseline", metric="acc_s",
                save_path="plot_heatmap_baseline.png",
                wandb_run=wandb_run,
            ),
        )
        if emissions is not None:
            stage_emissions.append(("Baseline heatmap", emissions))

        _, emissions = _run_tracked_stage(
            "alpha_heatmap",
            codecarbon_enabled,
            args.codecarbon_output_dir,
            args.codecarbon_project_name,
            wandb_run,
            lambda: plot_corruption_heatmap(
                rob_results, key=args.heatmap_alpha, metric="acc_s",
                save_path=f"plot_heatmap_alpha{str(args.heatmap_alpha).replace('.', '')}.png",
                wandb_run=wandb_run,
            ),
        )
        if emissions is not None:
            stage_emissions.append((f"Alpha heatmap ({args.heatmap_alpha})", emissions))

        _, emissions = _run_tracked_stage(
            "cascade_accuracy_gain_plot",
            codecarbon_enabled,
            args.codecarbon_output_dir,
            args.codecarbon_project_name,
            wandb_run,
            lambda: plot_cascade_accuracy_gain(
                rob_results, alphas,
                save_path="plot_cascade_accuracy_gain.png",
                wandb_run=wandb_run,
            ),
        )
        if emissions is not None:
            stage_emissions.append(("Cascade accuracy gain plot", emissions))

        _, emissions = _run_tracked_stage(
            "tau_accuracy_plot",
            codecarbon_enabled,
            args.codecarbon_output_dir,
            args.codecarbon_project_name,
            wandb_run,
            lambda: plot_tau_accuracy_tradeoff(
                rob_results, alphas, tau_values,
                save_path="plot_tau_accuracy_tradeoff.png",
                wandb_run=wandb_run,
            ),
        )
        if emissions is not None:
            stage_emissions.append(("Tau accuracy plot", emissions))

        if stage_emissions:
            _plot_stage_emissions(
                stage_emissions,
                save_path="plot_stage_emissions.png",
                wandb_run=wandb_run,
            )
    finally:
        if wandb_run is not None:
            if stage_emissions:
                wandb_run.summary["codecarbon/total_emissions_kg"] = float(sum(value for _, value in stage_emissions))
            wandb_run.finish()