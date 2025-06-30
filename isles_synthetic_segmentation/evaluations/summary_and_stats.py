import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

# Set up paths
PREDICTIONS_DIR = '../test/predictions'
OUTPUT_DIR = '.'

# Find all experiment result files
result_files = glob.glob(os.path.join(PREDICTIONS_DIR, '*/results.csv'))

# Aggregate all results
all_results = []
for file in result_files:
    exp_name = os.path.basename(os.path.dirname(file))
    df = pd.read_csv(file)
    df['experiment'] = exp_name
    all_results.append(df)

data = pd.concat(all_results, ignore_index=True)

# Metrics to analyze
metrics = ['dice', 'precision', 'recall']

# Summary table
summary = data.groupby('experiment')[metrics].agg(['mean', 'std'])
summary.columns = ['_'.join(col) for col in summary.columns]
summary.reset_index(inplace=True)
summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_table.csv'), index=False)
print('Summary table:')
print(summary)

# Boxplots
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='experiment', y=metric, data=data)
    plt.title(f'{metric.capitalize()} by Experiment')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Experiment')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{metric}_boxplot.png'))
    plt.close()

# Kruskal-Wallis test
stat_results = []
for metric in metrics:
    groups = [group[metric].dropna().values for name, group in data.groupby('experiment')]
    stat, p = kruskal(*groups)
    stat_results.append({'metric': metric, 'test': 'Kruskal-Wallis', 'statistic': stat, 'p_value': p})
    print(f'Kruskal-Wallis test for {metric}: statistic={stat:.4f}, p-value={p:.4e}')

# Pairwise Mann-Whitney U tests
from itertools import combinations
for metric in metrics:
    exps = data['experiment'].unique()
    for exp1, exp2 in combinations(exps, 2):
        vals1 = data[data['experiment'] == exp1][metric].dropna().values
        vals2 = data[data['experiment'] == exp2][metric].dropna().values
        if len(vals1) > 0 and len(vals2) > 0:
            stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
            stat_results.append({'metric': metric, 'test': f'Mann-Whitney {exp1} vs {exp2}', 'statistic': stat, 'p_value': p})
            print(f'Mann-Whitney U for {metric} ({exp1} vs {exp2}): statistic={stat:.4f}, p-value={p:.4e}')

# Save statistical test results
stat_df = pd.DataFrame(stat_results)
stat_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests.csv'), index=False)
print('Saved summary_table.csv, boxplots, and statistical_tests.csv in evaluations/') 