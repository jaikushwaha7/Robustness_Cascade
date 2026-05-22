"""
CIFAR-100 Gatekeeper Cascade Evaluation Analysis
==================================================
Fully analyses eval_results_cifar100.pkl:
  - Scalar metrics table (s_o, s_d, acc_s, acc_l) per alpha
  - Confidence distributions (correct vs incorrect) per alpha
  - Density overlap plots
  - Deferral curve: real vs random vs ideal (accs_real/rand/ideal vs deferral_ratios)
  - Summary heatmap
  - Tradeoff scatter: s_o vs s_d vs acc_s
  - AUC under deferral curve (real vs random gap)
"""

import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy import integrate
import warnings
warnings.filterwarnings("ignore")

PKL  = 'results/eval_results_cifar100.pkl'
OUT  = 'plots/cifar100/'

with open(PKL, 'rb') as f:
    R = pickle.load(f)

KEYS   = list(R.keys())                        # ['baseline', 0.9, 0.7, 0.5, 0.3, 0.1]
ALPHAS = [k for k in KEYS if k != 'baseline']
LABELS = {k: ('Baseline' if k == 'baseline' else f'α={k}') for k in KEYS}
COLORS = {
    'baseline': '#E24B4A',
    0.9: '#AFA9EC', 0.7: '#85B7EB',
    0.5: '#5DCAA5', 0.3: '#EF9F27', 0.1: '#534AB7',
}
LSDASH = {k: ('--' if k == 'baseline' else '-') for k in KEYS}

def r(key, field):
    return R[key][field]

# ─── compute AUC under real deferral curve (area above random) ────────────────
def auc_above_random(key):
    dr   = r(key, 'deferral_ratios')
    real = r(key, 'accs_real')
    rand = r(key, 'accs_rand')
    order = np.argsort(dr)
    return float(np.trapezoid(real[order] - rand[order], dr[order]))

# ══════════════════════════════════════════════════════════════════
# FIG 1 ─ Scalar metrics summary table (bar chart)
# ══════════════════════════════════════════════════════════════════
metrics_scalar = ['s_o', 's_d', 'acc_s']
titles_scalar  = ['Distributional Overlap s_o\n(↓ better)', 
                  'Deferral Score s_d\n(↑ better)',
                  'Small Model Accuracy acc_s\n(↑ standalone)']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CIFAR-100 Gatekeeper Cascade — Scalar Metrics by α',
             fontsize=13, fontweight='bold')

for ax, metric, title in zip(axes, metrics_scalar, titles_scalar):
    vals = [r(k, metric) for k in KEYS]
    bars = ax.bar(range(len(KEYS)), vals,
                  color=[COLORS[k] for k in KEYS],
                  edgecolor='black', linewidth=1.2, width=0.6)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=9, rotation=15)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, max(vals) * 1.18)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_scalar_metrics.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig1_scalar_metrics.png')

# ══════════════════════════════════════════════════════════════════
# FIG 2 ─ Confidence distributions: correct vs incorrect, all alphas
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('CIFAR-100 — Confidence Distributions: Correct vs Incorrect per α',
             fontsize=13, fontweight='bold')

for ax, key in zip(axes.flat, KEYS):
    cc = r(key, 'correct_confs')
    ic = r(key, 'incorrect_confs')

    bins = np.linspace(0, 1, 50)
    ax.hist(ic, bins=bins, density=True, alpha=0.55, color='#E24B4A', label='Incorrect')
    ax.hist(cc, bins=bins, density=True, alpha=0.55, color='#5DCAA5', label='Correct')

    # KDE lines
    for confs, col in [(cc, '#1a6e4a'), (ic, '#8b1a1a')]:
        if len(confs) > 2:
            kde = gaussian_kde(confs, bw_method=0.15)
            xs  = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=col, linewidth=2)

    ax.set_title(f'{LABELS[key]}\ns_o={r(key,"s_o"):.4f}  acc_s={r(key,"acc_s"):.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Confidence', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/fig2_confidence_distributions.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig2_confidence_distributions.png')

# ══════════════════════════════════════════════════════════════════
# FIG 3 ─ Density overlap (dens_c vs dens_i) — all alphas overlaid
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('CIFAR-100 — KDE Density: Correct (solid) vs Incorrect (dashed)',
             fontsize=13, fontweight='bold')

for ax, key in zip(axes.flat, KEYS):
    grid   = r(key, 'grid')
    dens_c = r(key, 'dens_c')
    dens_i = r(key, 'dens_i')

    ax.plot(grid, dens_c, color='#5DCAA5', linewidth=2.2, label='Correct')
    ax.plot(grid, dens_i, color='#E24B4A', linewidth=2.2, linestyle='--', label='Incorrect')

    # Shade overlap
    overlap = np.minimum(dens_c, dens_i)
    ax.fill_between(grid, overlap, alpha=0.35, color='#EF9F27', label='Overlap')

    ax.set_title(f'{LABELS[key]}  |  s_o={r(key,"s_o"):.4f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Confidence', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_density_overlap.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig3_density_overlap.png')

# ══════════════════════════════════════════════════════════════════
# FIG 4 ─ Deferral curves: real vs random vs ideal (all alphas)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('CIFAR-100 — Deferral Curve: Real vs Random vs Ideal',
             fontsize=13, fontweight='bold')

for ax, key in zip(axes.flat, KEYS):
    dr    = r(key, 'deferral_ratios')
    real  = r(key, 'accs_real')
    rand  = r(key, 'accs_rand')
    ideal = r(key, 'accs_ideal')
    order = np.argsort(dr)

    ax.plot(dr[order], real[order],  color=COLORS[key], linewidth=2.2, label='Real')
    ax.plot(dr[order], rand[order],  color='#888888',   linewidth=1.5,
            linestyle='--', label='Random')
    ax.plot(dr[order], ideal[order], color='#2c7bb6',   linewidth=1.5,
            linestyle=':', label='Ideal')

    # Shade area: real minus random
    ax.fill_between(dr[order], rand[order], real[order],
                    where=(real[order] >= rand[order]),
                    alpha=0.2, color=COLORS[key], label='Gain over random')

    auc = auc_above_random(key)
    ax.set_title(f'{LABELS[key]}\ns_d={r(key,"s_d"):.4f}  AUC_gain={auc:.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Deferral ratio', fontsize=9)
    ax.set_ylabel('Cascade accuracy', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_deferral_curves.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig4_deferral_curves.png')

# ══════════════════════════════════════════════════════════════════
# FIG 5 ─ All configs on ONE deferral curve (real) — comparison
# ══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('CIFAR-100 — Deferral Curves Comparison Across All α',
             fontsize=13, fontweight='bold')

for key in KEYS:
    dr    = r(key, 'deferral_ratios')
    real  = r(key, 'accs_real')
    rand  = r(key, 'accs_rand')
    order = np.argsort(dr)
    ax1.plot(dr[order], real[order], color=COLORS[key], linestyle=LSDASH[key],
             linewidth=2, label=LABELS[key])
    ax2.plot(dr[order], rand[order], color=COLORS[key], linestyle=LSDASH[key],
             linewidth=2, label=LABELS[key])

for ax, title in zip([ax1, ax2],
                     ['Real Cascade Accuracy vs Deferral',
                      'Random Deferral Accuracy vs Deferral']):
    ax.set_xlabel('Deferral ratio →', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cascade accuracy ↑', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f'{OUT}/fig5_deferral_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig5_deferral_comparison.png')

# ══════════════════════════════════════════════════════════════════
# FIG 6 ─ Summary heatmap: all metrics × all alphas
# ══════════════════════════════════════════════════════════════════
metrics_hm = ['acc_s', 's_o', 's_d', 
               'mean_correct_conf', 'mean_incorrect_conf', 
               'conf_separation', 'auc_gain',
               'n_correct', 'n_incorrect', 'deferral_at_50pct']

rows = []
for key in KEYS:
    dr   = r(key, 'deferral_ratios')
    real = r(key, 'accs_real')
    order = np.argsort(dr)
    # Find accuracy at 50% deferral
    idx50 = np.argmin(np.abs(dr[order] - 0.5))
    acc_50 = real[order][idx50]

    rows.append({
        'Config':               LABELS[key],
        'acc_s':                r(key, 'acc_s'),
        'acc_l':                r(key, 'acc_l'),
        's_o':                  r(key, 's_o'),
        's_d':                  r(key, 's_d'),
        'mean_correct_conf':    r(key, 'correct_confs').mean(),
        'mean_incorrect_conf':  r(key, 'incorrect_confs').mean(),
        'conf_separation':      r(key, 'correct_confs').mean() - r(key, 'incorrect_confs').mean(),
        'auc_gain':             auc_above_random(key),
        'n_correct':            len(r(key, 'correct_confs')),
        'n_incorrect':          len(r(key, 'incorrect_confs')),
        'acc_at_50pct_deferral': acc_50,
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(f'{OUT}/cifar100_summary.csv', index=False)
print('✓ cifar100_summary.csv')

# Numeric-only heatmap
hm_cols = ['acc_s', 's_o', 's_d', 'mean_correct_conf', 'mean_incorrect_conf',
           'conf_separation', 'auc_gain', 'acc_at_50pct_deferral']
hm_labels = ['acc_s', 's_o ↓', 's_d ↑', 'μ conf correct', 'μ conf incorrect',
             'conf separation', 'AUC gain', 'acc@50% defer']

hm_data = summary_df[hm_cols].values   # shape (6, 8)

# Normalise each col 0→1 for display
hm_norm = (hm_data - hm_data.min(axis=0)) / (np.ptp(hm_data, axis=0) + 1e-9)
# Invert s_o column (lower is better → higher display)
so_idx = hm_cols.index('s_o')
hm_norm[:, so_idx] = 1 - hm_norm[:, so_idx]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(hm_norm.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

ax.set_xticks(range(len(KEYS)))
ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=11)
ax.set_yticks(range(len(hm_labels)))
ax.set_yticklabels(hm_labels, fontsize=10)
ax.set_title('CIFAR-100 — Summary Heatmap (green = better, red = worse)',
             fontsize=12, fontweight='bold')

# Annotate with raw values
for i in range(len(KEYS)):
    for j in range(len(hm_cols)):
        ax.text(i, j, f'{hm_data[i, j]:.3f}',
                ha='center', va='center', fontsize=8, color='black')

plt.colorbar(im, ax=ax, fraction=0.02, label='Normalised performance')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_summary_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig6_summary_heatmap.png')

# ══════════════════════════════════════════════════════════════════
# FIG 7 ─ Tradeoff: s_o vs s_d vs acc_s
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('CIFAR-100 — Metric Tradeoffs Across α', fontsize=13, fontweight='bold')

pairs = [('s_o', 's_d',  's_o (overlap ↓)', 's_d (deferral ↑)'),
         ('s_o', 'acc_s','s_o (overlap ↓)', 'acc_s (small model acc)'),
         ('s_d', 'acc_s','s_d (deferral ↑)', 'acc_s (small model acc)')]

for ax, (xm, ym, xl, yl) in zip(axes, pairs):
    for key in KEYS:
        xv = r(key, xm)
        yv = r(key, ym)
        ax.scatter(xv, yv, color=COLORS[key], s=140, zorder=4,
                   marker='D' if key == 'baseline' else 'o',
                   edgecolors='black', linewidths=0.8)
        ax.annotate(LABELS[key], (xv, yv),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.set_xlabel(xl, fontsize=10, fontweight='bold')
    ax.set_ylabel(yl, fontsize=10, fontweight='bold')
    ax.set_title(f'{xl.split()[0]} vs {yl.split()[0]}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f'{OUT}/fig7_tradeoff_scatter.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig7_tradeoff_scatter.png')

# ══════════════════════════════════════════════════════════════════
# FIG 8 ─ Confidence distribution means + separation bar chart
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CIFAR-100 — Confidence Mean & Separation by α', fontsize=13, fontweight='bold')

x = np.arange(len(KEYS))
w = 0.35

ax = axes[0]
cc_means = [r(k, 'correct_confs').mean()   for k in KEYS]
ic_means = [r(k, 'incorrect_confs').mean() for k in KEYS]
ax.bar(x - w/2, cc_means, w, label='Correct',   color='#5DCAA5', edgecolor='black', linewidth=1)
ax.bar(x + w/2, ic_means, w, label='Incorrect', color='#E24B4A', edgecolor='black', linewidth=1)
for i, (cc, ic) in enumerate(zip(cc_means, ic_means)):
    ax.text(i - w/2, cc + 0.01, f'{cc:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.text(i + w/2, ic + 0.01, f'{ic:.3f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([LABELS[k] for k in KEYS], rotation=15)
ax.set_ylabel('Mean confidence', fontsize=10); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)
ax.set_title('Mean Confidence: Correct vs Incorrect', fontsize=11, fontweight='bold')

ax = axes[1]
separation = [r(k,'correct_confs').mean() - r(k,'incorrect_confs').mean() for k in KEYS]
bars = axes[1].bar(x, separation, color=[COLORS[k] for k in KEYS], edgecolor='black', linewidth=1.2)
for b, v in zip(bars, separation):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.4f}',
                 ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([LABELS[k] for k in KEYS], rotation=15)
ax.set_ylabel('Confidence separation (correct − incorrect)', fontsize=9)
ax.set_title('Confidence Separation\n(↑ better gating signal)', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig(f'{OUT}/fig8_confidence_separation.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig8_confidence_separation.png')

# ══════════════════════════════════════════════════════════════════
# FIG 9 ─ AUC gain above random per alpha
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
aucs = [auc_above_random(k) for k in KEYS]
bars = ax.bar(range(len(KEYS)), aucs, color=[COLORS[k] for k in KEYS],
              edgecolor='black', linewidth=1.2, width=0.6)
for b, v in zip(bars, aucs):
    ax.text(b.get_x()+b.get_width()/2, v+0.0005, f'{v:.4f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(KEYS)))
ax.set_xticklabels([LABELS[k] for k in KEYS], fontsize=10)
ax.set_ylabel('AUC gain above random deferral', fontsize=10)
ax.set_title('CIFAR-100 — Deferral Quality: AUC Gain Above Random Baseline\n(↑ better)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_auc_gain.png', dpi=150, bbox_inches='tight')
plt.close(); print('✓ fig9_auc_gain.png')

# ══════════════════════════════════════════════════════════════════
# PRINT TABLES
# ══════════════════════════════════════════════════════════════════
SEP = '═' * 110

print(f'\n{SEP}')
print('CIFAR-100 GATEKEEPER CASCADE — COMPLETE RESULTS TABLE')
print(SEP)

cols = ['Config', 'acc_s', 'acc_l', 's_o', 's_d',
        'mean_correct_conf', 'mean_incorrect_conf',
        'conf_separation', 'auc_gain',
        'n_correct', 'n_incorrect', 'acc_at_50pct_deferral']

print('\n' + summary_df[cols].to_string(index=False))

print(f'\n{SEP}')
print('DELTA vs BASELINE')
print(SEP)

base = summary_df[summary_df['Config'] == 'Baseline'].iloc[0]
num_cols = ['acc_s', 's_o', 's_d', 'conf_separation', 'auc_gain', 'acc_at_50pct_deferral']
print(f"\n{'Config':<12}", end='')
for c in num_cols: print(f"{c:>22}", end='')
print()
print('─' * 110)
for _, row in summary_df.iterrows():
    print(f"{row['Config']:<12}", end='')
    for c in num_cols:
        delta = row[c] - base[c]
        mark = '' if row['Config'] == 'Baseline' else ('+' if delta >= 0 else '')
        val = f"{row[c]:.4f} ({mark}{delta:+.4f})" if row['Config'] != 'Baseline' else f"{row[c]:.4f}"
        print(f"{val:>22}", end='')
    print()

print(f'\n{SEP}')
print('KEY FINDINGS')
print(SEP)

best_sd   = summary_df.loc[summary_df['s_d'].idxmax(), 'Config']
best_so   = summary_df.loc[summary_df['s_o'].idxmin(), 'Config']
best_sep  = summary_df.loc[summary_df['conf_separation'].idxmax(), 'Config']
best_auc  = summary_df.loc[summary_df['auc_gain'].idxmax(), 'Config']
best_50   = summary_df.loc[summary_df['acc_at_50pct_deferral'].idxmax(), 'Config']

print(f"""
  acc_l  (large model ceiling): {base['acc_l']:.4f}  — fixed, all configs share same M_L
  acc_s  (small model range):   {summary_df['acc_s'].max():.4f} (Baseline) → {summary_df['acc_s'].min():.4f} (α=0.1)

  Best s_d  (deferral score):   {best_sd:<12}  {summary_df.loc[summary_df['s_d'].idxmax(), 's_d']:.4f}
  Best s_o  (overlap, lowest):  {best_so:<12}  {summary_df.loc[summary_df['s_o'].idxmin(), 's_o']:.4f}
  Best conf separation:         {best_sep:<12}  {summary_df.loc[summary_df['conf_separation'].idxmax(), 'conf_separation']:.4f}
  Best AUC gain over random:    {best_auc:<12}  {summary_df.loc[summary_df['auc_gain'].idxmax(), 'auc_gain']:.4f}
  Best accuracy @ 50% deferral: {best_50:<12}  {summary_df.loc[summary_df['acc_at_50pct_deferral'].idxmax(), 'acc_at_50pct_deferral']:.4f}

  Large gap: acc_s={base['acc_s']:.4f} vs acc_l={base['acc_l']:.4f} → cascade gain potential = {base['acc_l']-base['acc_s']:.4f} ppt
""")

print('✓ All analysis complete.')