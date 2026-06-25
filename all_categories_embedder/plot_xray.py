"""Bar chart of xray/scissors pixel ROC-AUC for the 4 fine-tuning scopes
(pretrained / last-layer / full-FT / LoRA), our own DDAD-DA fine-tuning against
rohan's xray/2000 UNet, mse_cos AE. Output: results_eval_xray_mine/xray_finetune_scissors.png"""
import json, glob, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = 'results_eval_xray_mine'
METHODS = [('pretrained', 'pretrained\n(0 params)'),
           ('last_layer', 'last layer\n(1.1M)'),
           ('full',       'full FT\n(44.5M)'),
           ('lora',       'LoRA\n(0.5M)')]
COLORS = ['#7f7f7f', '#2ca02c', '#1f77b4', '#d62728']

def px(m, sigma='sigma_5.0'):
    fs = [f for f in glob.glob(f'{ROOT}/{m}/*.json') if not f.endswith('summary.json')]
    vals = [json.load(open(f)).get('averaged', {}).get(sigma, {}).get('px_roc_auc') for f in fs]
    vals = [v for v in vals if v is not None]
    return vals[0] if vals else None

vals = [px(m) for m, _ in METHODS]
labels = [lab for _, lab in METHODS]
fig, ax = plt.subplots(figsize=(7, 5))
xs = range(len(METHODS))
bars = ax.bar(xs, [v if v is not None else 0 for v in vals], color=COLORS, width=0.6)
for x, v in zip(xs, vals):
    if v is not None:
        ax.text(x, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('xray/scissors pixel ROC-AUC (smooth σ=5.0)', fontsize=11)
ax.set_title('X-ray fine-tuning (DDAD-DA vs xray/2000 UNet) judged by OUR detector', fontsize=12)
lo = min([v for v in vals if v is not None] + [0.9]) - 0.03
ax.set_ylim(max(0.5, lo), 1.0); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = f'{ROOT}/xray_finetune_scissors.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print('wrote', out, '| values:', {m: (round(v,3) if v is not None else None) for (m,_),v in zip(METHODS, vals)})
