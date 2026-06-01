"""
Evaluation Cifar10
============================

    casc_acc      — cascade accuracy at the operating alpha threshold
    gain_%        — % accuracy gain of cascade over standalone acc_s
    defer%        — deferral rate at operating threshold (scalar)
    Δacc_s        — change in acc_s vs baseline
    Δgain         — change in gain_% vs baseline
    mFP-MS        — high-confidence wrong predictions kept by MS / N
    mFP-Casc      — residual error rate of the cascade at operating point
    acc_casc      — alias for casc_acc (for CSV compatibility)
    n_correct     — number of correctly classified samples by MS
    n_incorrect   — number of incorrectly classified samples by MS
    N             — total test set size

CodeCarbon is tracked per stage (--codecarbon flag) and now also
writes a consolidated inference summary.
"""

import csv
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import logging
import re
from pathlib import Path
from typing import Any, Optional
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import SmallCNN, get_resnet18

try:
    import wandb
except ImportError:
    wandb = None

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None


_GPU_POWER_PATTERN = re.compile(r"Total GPU Power\s*:\s*([0-9.]+)\s*W")


def _sanitize_stage_name(stage_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_name).strip("_") or "stage"


class _CodeCarbonPowerCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.gpu_powers: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        match = _GPU_POWER_PATTERN.search(message)
        if match:
            self.gpu_powers.append(float(match.group(1)))


def _read_codecarbon_summary(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    def _to_float(field_name: str) -> float:
        try:
            return float(row.get(field_name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return {
        "Total Energy (kWh)": _to_float("energy_consumed"),
        "Total CO2 (g)":      _to_float("emissions") * 1000.0,
        "GPU Avg Power (W)":  _to_float("gpu_power"),
        "CPU Avg Power (W)":  _to_float("cpu_power"),
        "CO2 Rate (g/s)":     _to_float("emissions_rate") * 1000.0,
        "Est. Runtime (min)": _to_float("duration") / 60.0,
    }


def _start_codecarbon_tracker(enabled, output_dir, project_name, output_file):
    if not enabled or EmissionsTracker is None:
        return None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        output_file=output_file,
        save_to_file=True,
        save_to_api=False,
        save_to_logger=False,
    )
    tracker.start()
    return tracker


def _stop_codecarbon_tracker(tracker, wandb_run, metric_prefix, output_path):
    if tracker is None:
        return None
    emissions_kg = tracker.stop()
    metrics = _read_codecarbon_summary(output_path)
    if not metrics:
        metrics = {}
    metrics.setdefault("Total CO2 (g)",       float(emissions_kg) * 1000.0)
    metrics.setdefault("Total Energy (kWh)",  0.0)
    metrics.setdefault("GPU Avg Power (W)",   0.0)
    metrics.setdefault("CPU Avg Power (W)",   0.0)
    metrics.setdefault("CO2 Rate (g/s)",      0.0)
    metrics.setdefault("Est. Runtime (min)",  0.0)
    print(
        f"[{metric_prefix}] CodeCarbon: "
        f"energy={metrics['Total Energy (kWh)']:.6f} kWh | "
        f"co2={metrics['Total CO2 (g)']:.4f} g | "
        f"gpu_avg={metrics['GPU Avg Power (W)']:.1f} W | "
        f"cpu_avg={metrics['CPU Avg Power (W)']:.1f} W"
    )
    if wandb_run is not None:
        wandb_run.log({
            f"codecarbon/{metric_prefix}/total_energy_kwh":  metrics["Total Energy (kWh)"],
            f"codecarbon/{metric_prefix}/total_co2_g":       metrics["Total CO2 (g)"],
            f"codecarbon/{metric_prefix}/gpu_avg_power_w":   metrics["GPU Avg Power (W)"],
            f"codecarbon/{metric_prefix}/cpu_avg_power_w":   metrics["CPU Avg Power (W)"],
            f"codecarbon/{metric_prefix}/co2_rate_g_per_s":  metrics["CO2 Rate (g/s)"],
            f"codecarbon/{metric_prefix}/est_runtime_min":   metrics["Est. Runtime (min)"],
        })
        for k, v in metrics.items():
            wandb_run.summary[
                f"codecarbon/{metric_prefix}/{_sanitize_stage_name(k).lower()}"
            ] = float(v)
    return metrics


def _run_tracked_stage(stage_name, enabled, output_dir, project_name,
                       wandb_run, stage_fn):
    output_file = f"{_sanitize_stage_name(stage_name)}.csv"
    output_path = Path(output_dir) / output_file
    tracker = _start_codecarbon_tracker(enabled, output_dir, project_name, output_file)
    capture_handler = _CodeCarbonPowerCapture()
    cc_logger = logging.getLogger("codecarbon")
    prev_level = cc_logger.level
    cc_logger.setLevel(logging.INFO)
    cc_logger.addHandler(capture_handler)
    stage_error = result = None
    try:
        result = stage_fn()
    except Exception as exc:
        stage_error = exc
    finally:
        cc_logger.removeHandler(capture_handler)
        cc_logger.setLevel(prev_level)
        metrics = _stop_codecarbon_tracker(tracker, wandb_run, stage_name, output_path)
    if stage_error is not None:
        raise stage_error
    if metrics is None:
        metrics = {}
    metrics["Stage"] = stage_name
    metrics["GPU Max Power (W)"] = (
        max(capture_handler.gpu_powers)
        if capture_handler.gpu_powers
        else metrics.get("GPU Avg Power (W)", 0.0)
    )
    return result, metrics


def compute_accuracy(model, loader, device="cpu") -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return float(correct / total) if total > 0 else 0.0


def evaluate_ml_stage(model_l, val_loader, device="cpu", wandb_run=None):
    accuracy = compute_accuracy(model_l, val_loader, device=device)
    print(f"  ML → acc: {accuracy:.4f}")
    if wandb_run is not None:
        wandb_run.log({"eval/ml_acc": accuracy})
    return {"acc": accuracy}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_confidence_scores(model_s, loader, device="cpu"):
    model_s.eval()
    correct_confs, incorrect_confs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs = torch.softmax(model_s(images), dim=-1)
            conf, preds = probs.max(dim=-1)
            mask = preds == labels
            correct_confs.extend(conf[mask].cpu().numpy())
            incorrect_confs.extend(conf[~mask].cpu().numpy())
    return np.array(correct_confs, dtype=np.float32), \
           np.array(incorrect_confs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC: Distributional Overlap (s_o)
# ─────────────────────────────────────────────────────────────────────────────

def compute_distributional_overlap(correct_confs, incorrect_confs, n_grid=1000):
    from sklearn.neighbors import KernelDensity
    grid = np.linspace(0, 1, n_grid).reshape(-1, 1)
    kde_c = KernelDensity(bandwidth=0.05, kernel='gaussian').fit(correct_confs.reshape(-1, 1))
    kde_i = KernelDensity(bandwidth=0.05, kernel='gaussian').fit(incorrect_confs.reshape(-1, 1))
    dens_c = np.exp(kde_c.score_samples(grid))
    dens_i = np.exp(kde_i.score_samples(grid))
    dens_c /= dens_c.sum() * (1 / n_grid)
    dens_i /= dens_i.sum() * (1 / n_grid)
    s_o = np.minimum(dens_c, dens_i).sum() * (1 / n_grid)
    return s_o, dens_c, dens_i, grid.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# METRIC: Deferral Performance (s_d)
# ─────────────────────────────────────────────────────────────────────────────

def compute_deferral_performance(model_s, model_l, loader,
                                 tau_values=None, device="cpu"):
    if tau_values is None:
        tau_values = np.linspace(0.0, 1.0, 100)
    model_s.eval()
    model_l.eval()
    all_confs, all_preds_s, all_preds_l, all_labels = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs_s = torch.softmax(model_s(images), dim=-1)
            conf_s, pred_s = probs_s.max(dim=-1)
            pred_l = model_l(images).argmax(dim=-1)
            all_confs.append(conf_s.cpu())
            all_preds_s.append(pred_s.cpu())
            all_preds_l.append(pred_l.cpu())
            all_labels.append(labels)
    all_confs   = torch.cat(all_confs).numpy()
    all_preds_s = torch.cat(all_preds_s).numpy()
    all_preds_l = torch.cat(all_preds_l).numpy()
    all_labels  = torch.cat(all_labels).numpy()
    n = len(all_labels)
    acc_s = (all_preds_s == all_labels).mean()
    acc_l = (all_preds_l == all_labels).mean()

    deferral_ratios, joint_accs_real = [], []
    for tau in tau_values:
        defer_mask = all_confs < tau
        joint_preds = all_preds_s.copy()
        joint_preds[defer_mask] = all_preds_l[defer_mask]
        deferral_ratios.append(defer_mask.mean())
        joint_accs_real.append((joint_preds == all_labels).mean())

    deferral_ratios = np.array(deferral_ratios)
    joint_accs_real = np.array(joint_accs_real)
    sort_idx        = np.argsort(deferral_ratios)
    deferral_ratios = deferral_ratios[sort_idx]
    joint_accs_real = joint_accs_real[sort_idx]

    joint_accs_rand  = acc_s + (acc_l - acc_s) * deferral_ratios
    error_rate_s     = max(1 - acc_s, 1e-6)
    joint_accs_ideal = np.where(
        deferral_ratios <= error_rate_s,
        acc_s + (acc_l - acc_s) / error_rate_s * deferral_ratios,
        acc_l,
    )
    joint_accs_ideal = np.clip(joint_accs_ideal, acc_s, acc_l)

    uniform_grid = np.linspace(0, 1, 500)
    real_interp  = np.interp(uniform_grid, deferral_ratios, joint_accs_real)
    rand_interp  = np.interp(uniform_grid, deferral_ratios, joint_accs_rand)
    ideal_interp = np.interp(uniform_grid, deferral_ratios, joint_accs_ideal)
    area_real    = np.trapezoid(np.maximum(real_interp  - rand_interp, 0), uniform_grid)
    area_ideal   = np.trapezoid(np.maximum(ideal_interp - rand_interp, 0), uniform_grid)
    s_d = float(np.clip(area_real / area_ideal, 0.0, 1.0)) if area_ideal > 1e-8 else 0.0

    return (deferral_ratios, joint_accs_real, joint_accs_rand,
            joint_accs_ideal, s_d, acc_s, acc_l)


# ─────────────────────────────────────────────────────────────────────────────
# NEW ── compute_missing_metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_missing_metrics(stage_key: str,
                            correct_confs: np.ndarray,
                            incorrect_confs: np.ndarray,
                            deferral_ratios: np.ndarray,
                            accs_real: np.ndarray,
                            acc_s: float,
                            grid_1000: np.ndarray,
                            baseline_acc_s: float,
                            baseline_gain_pct: float) -> dict:
    """
    Computes the six metrics absent from the original eval script:

        casc_acc   — cascade accuracy at operating threshold
        gain_%     — (casc_acc - acc_s) / acc_s * 100
        defer%     — deferral ratio at operating threshold  (0–1)
        Δacc_s     — acc_s − baseline_acc_s
        Δgain      — gain_% − baseline_gain_%
        mFP-MS     — fraction of ALL samples: high-conf but wrong (kept by MS)
        mFP-Casc   — residual error of the cascade at op point (1 − casc_acc)
        n_correct  — len(correct_confs)
        n_incorrect— len(incorrect_confs)
        N          — total samples

    Parameters
    ----------
    stage_key       : "baseline" or float alpha (0.1 / 0.3 / 0.5 / 0.7 / 0.9)
    correct_confs   : confidence scores for MS-correct predictions
    incorrect_confs : confidence scores for MS-incorrect predictions
    deferral_ratios : x-axis of deferral curve (100 pts)
    accs_real       : realized cascade accuracy (100 pts, paired with deferral_ratios)
    acc_s           : standalone MS accuracy for this config
    grid_1000       : 1000-pt confidence threshold grid (from pkl)
    baseline_acc_s  : acc_s of the baseline config
    baseline_gain_pct : gain_% of the baseline config
    """
    N           = len(correct_confs) + len(incorrect_confs)
    n_correct   = len(correct_confs)
    n_incorrect = len(incorrect_confs)

    # ── Subsampled 100-pt grid (confidence thresholds) ──────────────────────
    grid_sub = grid_1000[::10][:100]

    # ── Operating point ──────────────────────────────────────────────────────
    if stage_key == "baseline":
        # Baseline has no fixed alpha — use first point where cascade beats acc_s
        above  = np.where(accs_real > acc_s)[0]
        op_idx = int(above[0]) if len(above) else 0
        tau    = float(grid_sub[op_idx])
    else:
        tau    = float(stage_key)
        op_idx = int(np.argmin(np.abs(grid_sub - tau)))

    casc_acc   = float(accs_real[op_idx])
    defer_pct  = float(deferral_ratios[op_idx])   # stored as 0–1 fraction

    # ── gain_% ───────────────────────────────────────────────────────────────
    gain_pct   = (casc_acc - acc_s) / acc_s * 100 if acc_s > 0 else 0.0

    # ── Δ metrics vs baseline ─────────────────────────────────────────────────
    delta_acc_s = acc_s - baseline_acc_s
    delta_gain  = gain_pct - baseline_gain_pct

    # ── mFP-MS ────────────────────────────────────────────────────────────────
    # High-confidence wrong predictions that MS keeps (conf ≥ tau) / N
    # These are the "false positives" — the small model is confident but wrong
    mfp_ms = float(np.sum(incorrect_confs >= tau) / N) if N > 0 else 0.0

    # ── mFP-Casc ─────────────────────────────────────────────────────────────
    # Residual error of cascade at operating point = 1 - casc_acc
    # Represents wrong predictions that the cascade system still makes after deferral
    mfp_casc = 1.0 - casc_acc

    return {
        "casc_acc":   round(casc_acc,   6),
        "acc_casc":   round(casc_acc,   6),   # alias for CSV compatibility
        "gain_%":     round(gain_pct,   4),
        "defer%":     round(defer_pct,  6),   # fraction 0–1
        "Δacc_s":     round(delta_acc_s, 6),
        "Δgain":      round(delta_gain,  4),
        "mFP-MS":     round(mfp_ms,     6),
        "mFP-Casc":   round(mfp_casc,   6),
        "n_correct":  n_correct,
        "n_incorrect": n_incorrect,
        "N":          N,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE STAGE EVALUATION  (patched to include missing metrics)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cascade_stage(model_s_pretrained, model_l, val_loader,
                           stage_key, num_classes=10, device="cpu",
                           wandb_run=None,
                           baseline_acc_s=None,
                           baseline_gain_pct=None,
                           codecarbon_enabled=False,
                           codecarbon_output_dir="codecarbon",
                           codecarbon_project_name="Robustness_Cascade"):
    if stage_key == "baseline":
        stage_name     = "MS"
        checkpoint     = "models/model_s_pretrained.pth"
        wandb_prefix   = "baseline"
    else:
        stage_name     = f"GK_alpha_{stage_key}"
        checkpoint     = f"models/model_s_gk_alpha{stage_key}.pth"
        wandb_prefix   = f"alpha_{stage_key}"

    print(f"Evaluating {stage_name}...")
    model_s = SmallCNN(num_classes=num_classes).to(device)
    model_s.load_state_dict(torch.load(checkpoint, map_location=device))

    def _tracked_inference():
        correct_confs, incorrect_confs = collect_confidence_scores(
            model_s, val_loader, device)
        deferral_ratios, accs_real, accs_rand, accs_ideal, s_d, acc_s, acc_l = compute_deferral_performance(
            model_s, model_l, val_loader, device=device)
        return {
            "correct_confs": correct_confs,
            "incorrect_confs": incorrect_confs,
            "deferral_ratios": deferral_ratios,
            "accs_real": accs_real,
            "accs_rand": accs_rand,
            "accs_ideal": accs_ideal,
            "s_d": s_d,
            "acc_s": float(acc_s),
            "acc_l": float(acc_l),
        }

    tracked_payload, tracked_metrics = _run_tracked_stage(
        stage_name,
        codecarbon_enabled,
        codecarbon_output_dir,
        codecarbon_project_name,
        wandb_run,
        _tracked_inference,
    )

    # Post-processing: KDE and derived metrics are intentionally untracked.
    correct_confs = tracked_payload["correct_confs"]
    incorrect_confs = tracked_payload["incorrect_confs"]
    deferral_ratios = tracked_payload["deferral_ratios"]
    accs_real = tracked_payload["accs_real"]
    accs_rand = tracked_payload["accs_rand"]
    accs_ideal = tracked_payload["accs_ideal"]
    s_d = tracked_payload["s_d"]
    acc_s = tracked_payload["acc_s"]
    acc_l = tracked_payload["acc_l"]

    s_o, dens_c, dens_i, grid = compute_distributional_overlap(
        correct_confs, incorrect_confs)

    # Baseline references must be known before computing deltas for alpha configs.
    # Pass baseline_acc_s=None on first call; the caller sets it after baseline run.
    _baseline_acc_s = baseline_acc_s if baseline_acc_s is not None else float(acc_s)
    _baseline_gain = baseline_gain_pct if baseline_gain_pct is not None else 0.0

    missing = compute_missing_metrics(
        stage_key=stage_key,
        correct_confs=correct_confs,
        incorrect_confs=incorrect_confs,
        deferral_ratios=deferral_ratios,
        accs_real=accs_real,
        acc_s=float(acc_s),
        grid_1000=grid,
        baseline_acc_s=_baseline_acc_s,
        baseline_gain_pct=_baseline_gain,
    )

    results = {
        # Original
        "s_o": s_o, "s_d": s_d, "acc_s": float(acc_s), "acc_l": float(acc_l),
        "correct_confs": correct_confs, "incorrect_confs": incorrect_confs,
        "dens_c": dens_c, "dens_i": dens_i, "grid": grid,
        "deferral_ratios": deferral_ratios,
        "accs_real": accs_real, "accs_rand": accs_rand, "accs_ideal": accs_ideal,
        # NEW
        **missing,
    }

    print(f"  {stage_name} → s_o:{s_o:.4f} | s_d:{s_d:.4f} | acc_s:{acc_s:.4f} | "
          f"casc_acc:{missing['casc_acc']:.4f} | gain:{missing['gain_%']:.2f}% | "
          f"defer:{missing['defer%']*100:.2f}% | mFP-MS:{missing['mFP-MS']:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            # Original
            f"eval/{wandb_prefix}/s_o":    s_o,
            f"eval/{wandb_prefix}/s_d":    s_d,
            f"eval/{wandb_prefix}/acc_s":  float(acc_s),
            f"eval/{wandb_prefix}/acc_l":  float(acc_l),
            # NEW
            f"eval/{wandb_prefix}/casc_acc":   missing["casc_acc"],
            f"eval/{wandb_prefix}/gain_pct":   missing["gain_%"],
            f"eval/{wandb_prefix}/defer_pct":  missing["defer%"] * 100,
            f"eval/{wandb_prefix}/delta_acc_s":missing["Δacc_s"],
            f"eval/{wandb_prefix}/delta_gain": missing["Δgain"],
            f"eval/{wandb_prefix}/mfp_ms":     missing["mFP-MS"],
            f"eval/{wandb_prefix}/mfp_casc":   missing["mFP-Casc"],
            f"eval/{wandb_prefix}/n_correct":  missing["n_correct"],
            f"eval/{wandb_prefix}/n_incorrect":missing["n_incorrect"],
        })

    return stage_name, results, tracked_metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project",          type=str, default="Robustness_Cascade")
    parser.add_argument("--wandb-entity",            type=str, default=None)
    parser.add_argument("--wandb-name",              type=str, default=None)
    parser.add_argument("--wandb-mode",              type=str, default="online")
    parser.add_argument("--no-wandb",                action="store_true")
    parser.add_argument("--codecarbon",              action="store_true")
    parser.add_argument("--codecarbon-output-dir",   type=str, default="codecarbon")
    parser.add_argument("--codecarbon-project-name", type=str, default="Robustness_Cascade")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb_run = None
    if not args.no_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={"device": device, "alphas": [0.9, 0.7, 0.5, 0.3, 0.1]},
        )

    codecarbon_enabled = args.codecarbon and EmissionsTracker is not None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_set   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform)
    val_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    model_l = get_resnet18(num_classes=10).to(device)
    model_l.load_state_dict(torch.load("models/model_l_pretrained.pth", map_location=device))
    model_l.eval()

    alphas = [0.9, 0.7, 0.5, 0.3, 0.1]
    results: dict = {}
    stage_metrics_by_name: dict = {}

    # ── ML evaluation ────────────────────────────────────────────────────────
    _, ml_metrics = _run_tracked_stage(
        "ML", codecarbon_enabled,
        args.codecarbon_output_dir, args.codecarbon_project_name, wandb_run,
        lambda: evaluate_ml_stage(model_l, val_loader, device=device, wandb_run=wandb_run),
    )
    stage_metrics_by_name["ML"] = ml_metrics

    # ── Baseline evaluation ───────────────────────────────────────────────────
    ms_stage_name, ms_results, ms_metrics = evaluate_cascade_stage(
        model_l, model_l, val_loader, "baseline",
        num_classes=10, device=device, wandb_run=wandb_run,
        baseline_acc_s=None,     # will self-reference
        baseline_gain_pct=None,
        codecarbon_enabled=codecarbon_enabled,
        codecarbon_output_dir=args.codecarbon_output_dir,
        codecarbon_project_name=args.codecarbon_project_name,
    )
    results["baseline"] = ms_results
    stage_metrics_by_name[ms_stage_name] = ms_metrics

    # Store baseline references for delta computation
    _baseline_acc_s   = ms_results["acc_s"]
    _baseline_gain    = ms_results["gain_%"]

    # ── Alpha evaluations ────────────────────────────────────────────────────
    for alpha in alphas:
        stage_label = f"GK_alpha_{alpha}"
        alpha_stage_name, alpha_results, alpha_metrics = evaluate_cascade_stage(
            model_l, model_l, val_loader, alpha,
            num_classes=10, device=device, wandb_run=wandb_run,
            baseline_acc_s=_baseline_acc_s,
            baseline_gain_pct=_baseline_gain,
            codecarbon_enabled=codecarbon_enabled,
            codecarbon_output_dir=args.codecarbon_output_dir,
            codecarbon_project_name=args.codecarbon_project_name,
        )
        results[alpha] = alpha_results
        stage_metrics_by_name[alpha_stage_name] = alpha_metrics

    # ── Save pkl ──────────────────────────────────────────────────────────────
    with open("eval_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved → eval_results.pkl")

    # ── WandB artifact ────────────────────────────────────────────────────────
    if wandb_run is not None:
        artifact = wandb.Artifact("eval_results", type="evaluation")
        artifact.add_file("eval_results.pkl")
        wandb_run.log_artifact(artifact)

        # NEW: log full metrics table including missing metrics
        full_table = wandb.Table(columns=[
            "key", "s_o", "s_d", "acc_s", "acc_l",
            "casc_acc", "gain_%", "defer%",
            "Δacc_s", "Δgain", "mFP-MS", "mFP-Casc",
            "n_correct", "n_incorrect", "N",
        ])
        for key in ["baseline"] + alphas:
            r = results[key]
            full_table.add_data(
                str(key), r["s_o"], r["s_d"], r["acc_s"], r["acc_l"],
                r["casc_acc"], r["gain_%"], r["defer%"],
                r["Δacc_s"], r["Δgain"], r["mFP-MS"], r["mFP-Casc"],
                r["n_correct"], r["n_incorrect"], r["N"],
            )
        wandb_run.log({"eval/full_metrics_table": full_table})

        stage_order = ["GK_alpha_0.1", "GK_alpha_0.3", "GK_alpha_0.5",
                       "GK_alpha_0.7", "GK_alpha_0.9", "ML", "MS"]
        stage_rows = [stage_metrics_by_name[n] for n in stage_order if n in stage_metrics_by_name]
        if stage_rows:
            cc_table = wandb.Table(columns=[
                "Stage", "Total Energy (kWh)", "Total CO2 (g)",
                "GPU Avg Power (W)", "GPU Max Power (W)", "CPU Avg Power (W)",
                "CO2 Rate (g/s)", "Est. Runtime (min)",
            ])
            for row in stage_rows:
                cc_table.add_data(
                    row.get("Stage", ""),
                    row.get("Total Energy (kWh)", 0.0),
                    row.get("Total CO2 (g)", 0.0),
                    row.get("GPU Avg Power (W)", 0.0),
                    row.get("GPU Max Power (W)", 0.0),
                    row.get("CPU Avg Power (W)", 0.0),
                    row.get("CO2 Rate (g/s)", 0.0),
                    row.get("Est. Runtime (min)", 0.0),
                )
            wandb_run.log({"codecarbon/stage_metrics_table": cc_table})

        wandb_run.finish()

    # ── Console summary ───────────────────────────────────────────────────────
    print("\nFull metric summary (including new metrics):")
    print(f"{'Key':<12} {'s_o':>7} {'s_d':>7} {'acc_s':>7} {'casc_acc':>9} "
          f"{'gain_%':>7} {'defer%':>7} {'Δacc_s':>8} {'Δgain':>7} "
          f"{'mFP-MS':>8} {'mFP-Casc':>9}")
    print("-" * 95)
    for key in ["baseline"] + alphas:
        r = results[key]
        print(f"{str(key):<12} {r['s_o']:>7.4f} {r['s_d']:>7.4f} {r['acc_s']:>7.4f} "
              f"{r['casc_acc']:>9.4f} {r['gain_%']:>7.2f} {r['defer%']*100:>7.2f} "
              f"{r['Δacc_s']:>8.4f} {r['Δgain']:>7.2f} "
              f"{r['mFP-MS']:>8.4f} {r['mFP-Casc']:>9.4f}")