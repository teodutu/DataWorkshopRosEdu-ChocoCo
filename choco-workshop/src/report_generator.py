# Script to generate a Markdown report summarizing key results and figures
# Usage: python src/report_generator.py

import os
import pandas as pd
import glob

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../reports/figures')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '../reports/summary_report.md')

def get_metric_from_evaluate():
    # Try to extract metrics from evaluate.py output files
    metrics = {}
    eval_txt = os.path.join(PROCESSED_DATA_DIR, 'evaluation.txt')
    if os.path.exists(eval_txt):
        with open(eval_txt) as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    metrics[k.strip()] = v.strip()
    return metrics

def get_feature_importance():
    fi_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importances_model.csv')
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        return fi.head(5)
    return None

def get_permutation_importance():
    perm_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importances_permutation.csv')
    if os.path.exists(perm_path):
        perm = pd.read_csv(perm_path)
        return perm.head(5)
    return None

def get_figures():
    figs = glob.glob(os.path.join(FIGURES_DIR, '*.png'))
    return figs

def main():
    with open(REPORT_PATH, 'w') as f:
        f.write('# ChocoCo Sales Prediction Report\n\n')
        f.write('## Key Results\n')
        # Try to get metrics from evaluation
        metrics = get_metric_from_evaluate()
        if metrics:
            for k, v in metrics.items():
                f.write(f'- **{k}:** {v}\n')
        else:
            f.write('- (Run evaluate.py to generate metrics)\n')

        f.write('\n## Top Features (Model Importance)\n')
        fi = get_feature_importance()
        if fi is not None:
            f.write(fi.to_markdown(index=False))
            f.write('\n')
        else:
            f.write('- (No feature importance file found)\n')

        f.write('\n## Top Features (Permutation Importance)\n')
        perm = get_permutation_importance()
        if perm is not None:
            f.write(perm.to_markdown(index=False))
            f.write('\n')
        else:
            f.write('- (No permutation importance file found)\n')

        f.write('\n## Figures\n')
        figs = get_figures()
        for fig in figs:
            rel_path = os.path.relpath(fig, os.path.dirname(REPORT_PATH))
            f.write(f'![{os.path.basename(fig)}]({rel_path})\n')

        f.write('\n---\nReport generated automatically.\n')

if __name__ == '__main__':
    main()
