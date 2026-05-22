import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
# Using the filenames provided in the environment
FULL_FILES = {
    'brightness':                    './cifar10p_outputs/cifar10p_results_brightness.pkl',
    # 'partial_brightness':            './cifar10p_outputs/cifar10p_results_partial_brightness.pkl',
    'gaussian_blur':                 './cifar10p_outputs/cifar10p_results_gaussian_blur.pkl',
    # 'partial_gaussian_blur':         './cifar10p_outputs/cifar10p_results_partial_gaussian_blur.pkl',
    'gaussian_noise':                './cifar10p_outputs/cifar10p_results_gaussian_noise.pkl',
    # 'partial_gaussian_noise':        './cifar10p_outputs/cifar10p_results_partial_gaussian_noise.pkl',
    'motion_blur':                   './cifar10p_outputs/cifar10p_results_motion_blur.pkl',
    # 'partial_motion_blur':           './cifar10p_outputs/cifar10p_results_partial_motion_blur.pkl',
    'rotate':                        './cifar10p_outputs/cifar10p_results_rotate.pkl',
    # 'partial_rotate':                './cifar10p_outputs/cifar10p_results_partial_rotate.pkl',
    'translate':                     './cifar10p_outputs/cifar10p_results_translate.pkl',
    # 'partial_translate':             './cifar10p_outputs/cifar10p_results_partial_translate.pkl',
    'zoom_blur':                     './cifar10p_outputs/cifar10p_results_zoom_blur.pkl',
    # 'partial_zoom_blur':             './cifar10p_outputs/cifar10p_results_partial_zoom_blur.pkl',
    'scale':                         './cifar10p_outputs/cifar10p_results_scale.pkl',
    # 'partial_scale':                 './cifar10p_outputs/cifar10p_results_partial_scale.pkl',
    'speckle_noise':                 './cifar10p_outputs/cifar10p_results_speckle_noise.pkl',
    # 'partial_speckle_noise':         './cifar10p_outputs/cifar10p_results_partial_speckle_noise.pkl',
    'tilt':                          './cifar10p_outputs/cifar10p_results_tilt.pkl',
    # 'partial_tilt':                  './cifar10p_outputs/cifar10p_results_partial_tilt.pkl',
    'spatter':                       './cifar10p_outputs/cifar10p_results_spatter.pkl',
    # 'partial_spatter':               './cifar10p_outputs/cifar10p_results_partial_spatter.pkl',
    'shear':                         './cifar10p_outputs/cifar10p_results_shear.pkl',
    # 'partial_shear':                 './cifar10p_outputs/cifar10p_results_partial_shear.pkl',
    'shot_noise':                    './cifar10p_outputs/cifar10p_results_shot_noise.pkl',
    # 'partial_shot_noise':            './cifar10p_outputs/cifar10p_results_partial_shot_noise.pkl',
    'snow':                          './cifar10p_outputs/cifar10p_results_snow.pkl',
    # 'partial_snow':                  './cifar10p_outputs/cifar10p_results_partial_snow.pkl'
}

# ── load & normalise ───────────────────────────────────────────────────────────
DATA = {}   # DATA[pert][cfg] = record dict
for pert, path in FULL_FILES.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        inner = raw[pert]   # unwrap top-level perturbation key
        DATA[pert] = inner
    else:
        print(f"Warning: File {path} not found.")

if not DATA:
    print("Error: No data loaded. Please check file paths.")
    exit()

PERTS   = list(DATA.keys())
# Check available configs from the first perturbation
first_pert = PERTS[0]
CFGS    = list(DATA[first_pert].keys())          # ['baseline', 0.9, 0.7, 0.5, 0.3, 0.1]
ALPHAS  = [c for c in CFGS if c != 'baseline']
LABELS  = {c: ('Baseline' if c == 'baseline' else f'α={c}') for c in CFGS}

COLORS  = {
    'baseline': '#E24B4A',
    0.9: '#AFA9EC', 0.7: '#85B7EB', 0.5: '#5DCAA5',
    0.3: '#EF9F27', 0.1: '#534AB7',
}
LSDASH = {c: ('--' if c == 'baseline' else '-') for c in CFGS}

PERT_LABELS = {
    'brightness':     'Brightness',
    'gaussian_blur':  'Gaussian Blur',
    'gaussian_noise': 'Gaussian Noise',
    'motion_blur':    'Motion Blur',
    'rotate':         'Rotate',
    'translate':      'Translate',
    'zoom_blur':      'Zoom Blur',
    'scale':          'Scale',
    'speckle_noise':  'Speckle Noise',
    'tilt':           'Tilt',
    'spatter':        'Spatter',
    'shear':          'Shear',
    'shot_noise':     'Shot Noise',
    'snow':           'Snow',
}

# Output directory for images
OUT = './analysis_outputs'
os.makedirs(OUT, exist_ok=True)

def r(pert, cfg, key):
    return DATA[pert][cfg][key]

def steps(pert):
    # In some results, 'n_steps' is an int, in others it's inferred from the array length
    if 'n_steps' in DATA[pert]['baseline']:
        return DATA[pert]['baseline']['n_steps']
    return len(r(pert, 'baseline', 'acc_ms_per_step'))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — acc_ms & acc_cascade across perturbation steps
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(PERTS), 2, figsize=(14, 4*len(PERTS)))
fig.suptitle('Accuracy Across Perturbation Steps\nacc_ms (left) · acc_cascade (right)',
             fontsize=14, fontweight='bold', y=1.01)

for row, pert in enumerate(PERTS):
    n = steps(pert)
    xs = np.arange(n)
    for metric, col, ylabel in [
        ('acc_ms_per_step',      0, 'acc_ms'),
        ('acc_cascade_per_step', 1, 'cascade acc'),
    ]:
        ax = axes[row][col]
        for cfg in CFGS:
            vals = r(pert, cfg, metric)
            ax.plot(xs, vals, color=COLORS.get(cfg, '#333'), linestyle=LSDASH.get(cfg, '-'),
                    linewidth=1.8, label=LABELS[cfg])
        ax.set_title(f'{PERT_LABELS[pert]} — {ylabel}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Perturbation step', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_acc_per_step_complete.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Mean flip rate summary: grouped bar (pert × config)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Mean Flip Rates — Summary\n(lower = more stable predictions)',
             fontsize=13, fontweight='bold')

metrics_bar = [
    ('mfp_ms',            'mFP-MS (Small Model)', '#85B7EB'),
    ('mfp_cascade',       'mFP-Cascade',          '#5DCAA5'),
    ('mean_deferral_flip','Mean Deferral Flip',    '#EF9F27'),
]
x = np.arange(len(PERTS))
w = 0.13
offsets = np.linspace(-(len(CFGS)-1)/2*w, (len(CFGS)-1)/2*w, len(CFGS))

for ax, (metric, title, _) in zip(axes, metrics_bar):
    for i, cfg in enumerate(CFGS):
        vals = [r(pert, cfg, metric) for pert in PERTS]
        ax.bar(x + offsets[i], vals, width=w,
               color=COLORS.get(cfg, '#333'), label=LABELS[cfg],
               edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([PERT_LABELS[p].replace(' ', '\n') for p in PERTS], fontsize=9)
    ax.set_ylabel('Mean flip rate', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_mfp_summary_complete.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Mean metrics across ALL perturbation steps
# ══════════════════════════════════════════════════════════════════════════════
SEP = '═' * 95
print(f'\n{SEP}')
print('TABLE 1 — Mean metrics across ALL perturbation steps')
print(SEP)
hdr = f"{'Perturbation':<18} {'Config':<12} {'acc_ms':>8} {'acc_casc':>9} {'defer%':>8} {'mFP-MS':>8} {'mFP-Casc':>10}"
print(hdr); print('─'*95)

for pert in PERTS:
    for cfg in CFGS:
        acc_ms   = float(np.mean(r(pert, cfg, 'acc_ms_per_step')))
        acc_cas  = float(np.mean(r(pert, cfg, 'acc_cascade_per_step')))
        defer    = float(np.mean(r(pert, cfg, 'defer_rate_per_step'))) * 100
        mfp_ms   = float(r(pert, cfg, 'mfp_ms'))
        mfp_cas  = float(r(pert, cfg, 'mfp_cascade'))
        p_lbl    = PERT_LABELS[pert] if cfg == 'baseline' else ''
        print(f"{p_lbl:<18} {LABELS[cfg]:<12} {acc_ms:>8.4f} {acc_cas:>9.4f} {defer:>7.1f}% "
              f"{mfp_ms:>8.4f} {mfp_cas:>10.4f}")
    print('─'*95)