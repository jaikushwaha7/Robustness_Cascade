import os
import json
import torch
import pickle
import itertools
import numpy as np

from models import SmallCNN, get_resnet18
from evaluate import (
    collect_confidence_scores,
    compute_distributional_overlap,
    compute_deferral_performance
)

from augmix_eval import load_augmix_loader  # reuse your dataset loader


# ─────────────────────────────────────────────
# 1. EXPERIMENT GRID
# ─────────────────────────────────────────────

SEVERITY_GRID = [1, 3, 5]
WIDTH_GRID    = [2, 3, 5]
DEPTH_GRID    = [-1, 2, 4]
ALPHA_GRID    = [0.5, 1.0, 2.0]


EXPERIMENT_GRID = [
    {
        "severity": s,
        "width": w,
        "depth": d,
        "alpha": a
    }
    for s, w, d, a in itertools.product(
        SEVERITY_GRID,
        WIDTH_GRID,
        DEPTH_GRID,
        ALPHA_GRID
    )
]


# ─────────────────────────────────────────────
# 2. EVALUATION FUNCTION
# ─────────────────────────────────────────────

def evaluate_config(model_s, model_l, loader, device="cpu"):
    """
    Computes:
        s_o, s_d, acc_s, acc_casc
    """

    correct_confs, incorrect_confs = collect_confidence_scores(
        model_s, loader, device
    )

    s_o, _, _, _ = compute_distributional_overlap(
        correct_confs, incorrect_confs
    )

    (
        _, _, _, _, s_d,
        acc_s, acc_casc
    ) = compute_deferral_performance(
        model_s, model_l, loader, device=device
    )

    return {
        "s_o": s_o,
        "s_d": s_d,
        "acc_s": acc_s,
        "acc_casc": acc_casc
    }


# ─────────────────────────────────────────────
# 3. MAIN SWEEP LOOP
# ─────────────────────────────────────────────

def run_augmix_grid(
    alphas,
    model_s_ckpts,
    model_l_ckpt,
    device="cuda",
    save_path="augmix_grid_results.pkl"
):

    # Load large model once
    model_l = get_resnet18(num_classes=10).to(device)
    model_l.load_state_dict(torch.load(model_l_ckpt, map_location=device))
    model_l.eval()

    results = {}

    for alpha in alphas:

        print(f"\n==============================")
        print(f"Running Alpha Model: {alpha}")
        print(f"==============================")

        model_s = SmallCNN(num_classes=10).to(device)
        model_s.load_state_dict(
            torch.load(model_s_ckpts[alpha], map_location=device)
        )
        model_s.eval()

        results[alpha] = {}

        for cfg in EXPERIMENT_GRID:

            severity = cfg["severity"]
            width    = cfg["width"]
            depth    = cfg["depth"]
            alpha_a  = cfg["alpha"]

            loader = load_augmix_loader(
                severity=severity,
                batch_size=128
            )

            metrics = evaluate_config(
                model_s,
                model_l,
                loader,
                device=device
            )

            key = f"S{severity}_W{width}_D{depth}_A{alpha_a}"

            results[alpha][key] = {
                "config": cfg,
                "metrics": metrics
            }

            print(
                f"{key} | "
                f"s_o={metrics['s_o']:.4f} | "
                f"s_d={metrics['s_d']:.4f} | "
                f"acc_s={metrics['acc_s']:.4f} | "
                f"acc_casc={metrics['acc_casc']:.4f}"
            )

            # optional checkpoint save (safe for long runs)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)

    return results


# ─────────────────────────────────────────────
# 4. SUMMARY TABLE BUILDER
# ─────────────────────────────────────────────

def build_summary_table(results):
    """
    Converts nested dict → flat table for pandas / thesis.
    """

    rows = []

    for alpha, configs in results.items():
        for k, v in configs.items():

            cfg = v["config"]
            m   = v["metrics"]

            rows.append({
                "alpha_model": alpha,
                "severity": cfg["severity"],
                "width": cfg["width"],
                "depth": cfg["depth"],
                "aug_alpha": cfg["alpha"],

                "acc_s": m["acc_s"],
                "acc_casc": m["acc_casc"],
                "s_o": m["s_o"],
                "s_d": m["s_d"]
            })

    return rows


# ─────────────────────────────────────────────
# 5. RUN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # map alpha → checkpoint
    model_s_ckpts = {
        0.9: "model_s_gk_alpha0.9.pth",
        0.7: "model_s_gk_alpha0.7.pth",
        0.5: "model_s_gk_alpha0.5.pth",
        0.3: "model_s_gk_alpha0.3.pth",
        0.1: "model_s_gk_alpha0.1.pth",
        "baseline": "model_s_pretrained.pth"
    }

    alphas = [0.9, 0.7, 0.5, 0.3, 0.1]

    model_l_ckpt = "model_l_pretrained.pth"

    results = run_augmix_grid(
        alphas=alphas,
        model_s_ckpts=model_s_ckpts,
        model_l_ckpt=model_l_ckpt,
        device=device
    )

    # save final
    with open("augmix_grid_final.pkl", "wb") as f:
        pickle.dump(results, f)

    # flatten for analysis
    table = build_summary_table(results)

    with open("augmix_grid_table.json", "w") as f:
        json.dump(table, f, indent=2)

    print("\n✅ Saved all results:")
    print(" - augmix_grid_final.pkl")
    print(" - augmix_grid_table.json")