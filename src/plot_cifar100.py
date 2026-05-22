import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


# ─────────────────────────────────────────────
# PLOT 1: Distributional Overlap (Figure 3a)
# ─────────────────────────────────────────────

def plot_distributional_overlap(results, alpha_to_show=0.1,
                                  save_path="plots/cifar100/plot_overlap_cifar100.png",
                                  wandb_run=None):
    """
    Reproduces Figure 3a from the paper.
    Shows KDE of confidence scores for correct vs incorrect predictions,
    with the overlap area shaded.

    alpha_to_show: which alpha result to visualise (default: 0.1 = most aggressive)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    keys    = ["baseline", alpha_to_show]
    titles  = ["Baseline (no Gatekeeper)", f"Gatekeeper α={alpha_to_show}"]
    colors  = {"correct": "#4878CF", "incorrect": "#E8855A", "overlap": "#C97FB3"}

    for ax, key, title in zip(axes, keys, titles):
        r    = results[key]
        grid = r["grid"]
        dc   = r["dens_c"]
        di   = r["dens_i"]

        ax.plot(grid, dc, color=colors["correct"],   lw=2, label="Correct")
        ax.plot(grid, di, color=colors["incorrect"], lw=2, label="Incorrect")
        ax.fill_between(grid, np.minimum(dc, di),
                         alpha=0.4, color=colors["overlap"],
                         label=f"Overlap area $s_o$={r['s_o']:.3f}")

        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Density",    fontsize=12)
        ax.set_title(title,         fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("a) Distributional Overlap $s_o$ - CIFAR-100", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")
    if wandb_run is not None:
        wandb_run.log({"plot/overlap": wandb.Image(save_path)})


# ─────────────────────────────────────────────
# PLOT 2: Deferral Performance (Figure 3b)
# ─────────────────────────────────────────────

def plot_deferral_performance(results, alpha_to_show=0.1,
                               save_path="plots/cifar100/plot_deferral_cifar100.png",
                               wandb_run=None):
    """
    Reproduces Figure 3b from the paper.
    Shows ideal, random, and realized deferral curves
    with performance and headroom areas shaded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    keys   = ["baseline", alpha_to_show]
    titles = ["Baseline (no Gatekeeper)", f"Gatekeeper α={alpha_to_show}"]

    for ax, key, title in zip(axes, keys, titles):
        r  = results[key]
        dr = r["deferral_ratios"]
        ar = r["accs_real"]
        an = r["accs_rand"]
        ai = r["accs_ideal"]

        # ── Shaded areas ──
        # Blue: performance area (realized - random)
        ax.fill_between(dr, an, ar,
                         where=(ar >= an),
                         alpha=0.4, color="#6EB5E0",
                         label="Performance Area ($A_{perf}$)")

        # Green hatched: headroom (ideal - realized)
        ax.fill_between(dr, ar, ai,
                         where=(ai >= ar),
                         alpha=0.25, color="#5DBB63", hatch='///',
                         label="Headroom Area")

        # ── Curves ──
        ax.plot(dr, ai, color="#2CA02C", lw=2,
                linestyle='--', label="Ideal Deferral ($acc_{ideal}$)")
        ax.plot(dr, an, color="#D62728", lw=2,
                linestyle=':', label="Random Deferral ($acc_{rand}$)")
        ax.plot(dr, ar, color="black",   lw=2.5,
                label=f"Realized Deferral ($s_d$={r['s_d']:.3f})")

        # ── Dots: no deferral and full deferral ──
        ax.scatter([0], [r["acc_s"]], color="#BCBD22",
                   zorder=5, s=80, label=f"No deferral acc($M_S$)={r['acc_s']:.3f}")
        ax.scatter([1], [r["acc_l"]], color="#9467BD",
                   zorder=5, s=80, marker='s',
                   label=f"Full deferral acc($M_L$)={r['acc_l']:.3f}")

        ax.set_xlabel("Deferral Ratio",  fontsize=12)
        ax.set_ylabel("Joint Accuracy",  fontsize=12)
        ax.set_title(title,              fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("b) Deferral Performance $s_d$ - CIFAR-100", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")
    if wandb_run is not None:
        wandb_run.log({"plot/deferral": wandb.Image(save_path)})


# ─────────────────────────────────────────────
# PLOT 3: Alpha Sweep Summary (like Figure 4)
# Shows s_o, s_d, acc_s across all alpha values
# ─────────────────────────────────────────────

def plot_alpha_sweep(results, alphas,
                     save_path="plots/cifar100/plot_alpha_sweep_cifar100.png",
                     wandb_run=None):
    """
    Reproduces Figure 4 style: three subplots showing how
    s_o, s_d, and acc_s change as alpha varies.
    """
    keys   = ["baseline"] + alphas
    labels = ["Base"] + [str(a) for a in alphas]

    s_o_vals   = [results[k]["s_o"]   for k in keys]
    s_d_vals   = [results[k]["s_d"]   for k in keys]
    acc_s_vals = [results[k]["acc_s"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bar_color = "#5B9BD5"

    # s_o — lower is better
    axes[0].bar(labels, s_o_vals, color=bar_color, edgecolor='black', width=0.5)
    axes[0].set_title("Distributional Overlap $s_o$ ↓", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("$s_o$")
    axes[0].grid(axis='y', alpha=0.3)

    # s_d — higher is better
    axes[1].bar(labels, s_d_vals, color="#5DBB63", edgecolor='black', width=0.5)
    axes[1].set_title("Deferral Performance $s_d$ ↑", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("$s_d$")
    axes[1].grid(axis='y', alpha=0.3)

    # acc_s — context (not directly better/worse)
    axes[2].bar(labels, acc_s_vals, color="#E8855A", edgecolor='black', width=0.5)
    axes[2].set_title("Small Model Accuracy $acc(M_S)$", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Alpha")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle("CIFAR-100 — Gatekeeper Alpha Sweep", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_path}")
    if wandb_run is not None:
        table = wandb.Table(columns=["key", "s_o", "s_d", "acc_s"])
        for key in keys:
            table.add_data(str(key), results[key]["s_o"], results[key]["s_d"], results[key]["acc_s"])
        wandb_run.log({
            "plot/alpha_sweep": wandb.Image(save_path),
            "plot/alpha_sweep_table": table,
        })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot cascade results and optionally log figures to Weights & Biases.")
    parser.add_argument("--results-file", default="results/eval_results_cifar100.pkl")
    parser.add_argument("--wandb-project", type=str, default="Robustness_Cascade_CIFAR100")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    os.makedirs("plots/cifar100", exist_ok=True)

    with open(args.results_file, "rb") as f:
        results = pickle.load(f)

    alphas = [0.9, 0.7, 0.5, 0.3, 0.1]

    wandb_run = None
    if not args.no_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "results_file": args.results_file,
                "alphas": alphas,
                "plotting": True,
            },
        )
    elif not args.no_wandb:
        print("Weights & Biases is not installed, continuing without figure logging.")

    try:
        # Figure 3a style — distributional overlap
        plot_distributional_overlap(results, alpha_to_show=0.1,
                                                 save_path="plots/cifar100/plot_overlap_cifar100.png",
                                     wandb_run=wandb_run)

        # Figure 3b style — deferral performance
        plot_deferral_performance(results, alpha_to_show=0.1,
                                             save_path="plots/cifar100/plot_deferral_cifar100.png",
                                  wandb_run=wandb_run)

        # Figure 4 style — alpha sweep summary
        plot_alpha_sweep(results, alphas,
                                 save_path="plots/cifar100/plot_alpha_sweep_cifar100.png",
                         wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
