import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── paths for partial runs ───────────────────────────────────────────────────
PARTIAL_FILES = {
    'brightness':     './cifar10p_outputs/cifar10p_results_partial_brightness.pkl',
    'gaussian_noise': './cifar10p_outputs/cifar10p_results_partial_gaussian_noise.pkl',
    'motion_blur':    './cifar10p_outputs/cifar10p_results_partial_motion_blur.pkl',
    'rotate':         './cifar10p_outputs/cifar10p_results_partial_rotate.pkl',
}

# ── load & normalize ─────────────────────────────────────────────────────────
DATA = {}  # Will be structured as DATA[pert][cfg]
for pert, path in PARTIAL_FILES.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            # Partial files are structured as {cfg: metrics} directly
            DATA[pert] = pickle.load(f)
    else:
        print(f"Skipping {pert}: {path} not found.")

if not DATA:
    print("No data found. Check your file paths.")
    exit()

PERTS   = list(DATA.keys())
CFGS    = list(DATA[PERTS[0]].keys())  # e.g., ['baseline', 0.9, 0.7, 0.5, 0.3, 0.1]
LABELS  = {c: ('Baseline' if c == 'baseline' else f'α={c}') for c in CFGS}
COLORS  = {'baseline': '#E24B4A', 0.9: '#AFA9EC', 0.7: '#85B7EB', 0.5: '#5DCAA5', 0.3: '#EF9F27', 0.1: '#534AB7'}
LSDASH  = {c: ('--' if c == 'baseline' else '-') for c in CFGS}
PERT_LABS = {'brightness': 'Brightness', 'gaussian_noise': 'Gaussian Noise', 'motion_blur': 'Motion Blur', 'rotate': 'Rotate'}

OUT = './partial_analysis'
os.makedirs(OUT, exist_ok=True)

def r(pert, cfg, key): return DATA[pert][cfg][key]
def steps(pert): return DATA[pert]['baseline']['n_steps']

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD: Accuracy and Flip Rates
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(PERTS), 3, figsize=(18, 4*len(PERTS)))
fig.suptitle('Partial Run Analysis: Accuracy & Stability', fontsize=16, fontweight='bold', y=1.02)

for row, pert in enumerate(PERTS):
    n = steps(pert)
    xs_step = np.arange(n)
    xs_trans = np.arange(n - 1)
    
    # 1. Cascade Accuracy
    ax = axes[row][0]
    for cfg in CFGS:
        ax.plot(xs_step, r(pert, cfg, 'acc_cascade_per_step'), color=COLORS[cfg], 
                linestyle=LSDASH[cfg], label=LABELS[cfg])
    ax.set_title(f'{PERT_LABS[pert]} Accuracy')
    ax.legend(fontsize=7)

    # 2. Cascade Flip Rates
    ax = axes[row][1]
    for cfg in CFGS:
        ax.plot(xs_trans, r(pert, cfg, 'pred_flip_rates_cascade'), color=COLORS[cfg], 
                linestyle=LSDASH[cfg])
    ax.set_title(f'{PERT_LABS[pert]} Flip Rates')

    # 3. Deferral Rate
    ax = axes[row][2]
    for cfg in CFGS:
        ax.plot(xs_step, r(pert, cfg, 'defer_rate_per_step'), color=COLORS[cfg], 
                linestyle=LSDASH[cfg])
    ax.set_title(f'{PERT_LABS[pert]} Deferral %')

plt.tight_layout()
plt.savefig(f'{OUT}/partial_dashboard.png', dpi=150)
print(f"✓ Created {OUT}/partial_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'Perturbation':<15} {'Config':<10} {'Mean Acc':>10} {'mFP-Casc':>10} {'Defer%':>8}")
print("-" * 58)
for pert in PERTS:
    for cfg in CFGS:
        acc = np.mean(r(pert, cfg, 'acc_cascade_per_step'))
        mfp = r(pert, cfg, 'mfp_cascade')
        def_r = np.mean(r(pert, cfg, 'defer_rate_per_step')) * 100
        p_str = pert if cfg == 'baseline' else ""
        print(f"{p_str:<15} {LABELS[cfg]:<10} {acc:>10.4f} {mfp:>10.4f} {def_r:>7.1f}%")