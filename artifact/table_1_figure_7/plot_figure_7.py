import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os

indir = sys.argv[1]
outdir = sys.argv[2]
csvdir = sys.argv[3]

data_path = os.path.join(indir, 'table_1_figure_7_hs_8192.csv')
df = pd.read_csv(data_path)

df = df.rename(columns={'BL': 'num_tokens'})
pivot_df = df.pivot_table(
    index=['num_tokens', 'hidden_size'],
    columns='multimem_ctas',
    values='fused_arln_us',
    aggfunc='min'
).reset_index()

pivot_df.columns.name = None
pivot_df = pivot_df.rename(columns={
    0: 'num_tokens_per_gpu',
    **{c: f'fused_arln_us_ctas_{int(c)}' for c in pivot_df.columns if isinstance(c, (int, float)) and c != 0}
})
pivot_path = os.path.join(csvdir, 'figure_7.csv')
pivot_df.to_csv(pivot_path, index=False)

melted = pivot_df.melt(id_vars=['num_tokens', 'hidden_size'], var_name='ctas_col', value_name='fused_arln_us')
melted['num_ctas'] = melted['ctas_col'].str.extract(r'(\d+)').astype(float)
melted['num_ctas'] = melted['num_ctas'].fillna(0).astype(int)
melted['num_tokens_str'] = melted['num_tokens'].astype(str)
melted = melted.sort_values(['num_tokens', 'num_ctas'])
melted = melted[melted['num_ctas'] != 0]
fig, ax = plt.subplots(figsize=(10, 6))

sns.set_theme(style="whitegrid")
sns.set_context("paper")

sns.lineplot(
    data=melted,
    x='num_ctas',
    y='fused_arln_us',
    hue='num_tokens_str',
    marker='o',
    linewidth=2,
    palette='deep',
    markers=True,
    ax=ax
)

ax.set_xlabel("#SMs", fontsize=18, fontweight='bold')
ax.set_ylabel("Fused(AR, RMSNorm) Time (us)", fontsize=18, fontweight='bold')
ax.set_yscale('log', base=10)
ax.set_xticks(melted['num_ctas'].unique())
ax.set_xscale('log', base=2)
ax.legend(title='Num Tokens', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, title_fontsize=16)
ax.tick_params(axis='both', labelsize=20)
ax.grid(axis='y', linestyle='--', linewidth=1.0, alpha=0.5)

plt.tight_layout()
output_path = os.path.join(outdir, 'figure_7.pdf')
plt.savefig(output_path, bbox_inches='tight', dpi=1200)
plt.close()
