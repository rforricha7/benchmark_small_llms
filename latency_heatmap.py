#!/usr/bin/env python3
"""
Produce 2D heatmaps for latency (or cost) vs prompt/output token counts.

Creates a heatmap where:
  - X axis = prompt token count (prompt_eval_count)
  - Y axis = output token count (eval_count)
  - Color = mean latency (or cost) in each bin

Usage:
  python3 latency_heatmap.py --metric latency
  python3 latency_heatmap.py --metric cost --cost-per-token 0.000002

The script writes PNGs into the `heatmaps/` directory.
"""

import os
import ast
import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def _safe_parse_metadata(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            # try to coerce quotes
            return ast.literal_eval(str(s).replace("None", "null").replace("'", '"'))
        except Exception:
            return {}


def _ns_to_s(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    # heuristics: values > 1e6 likely nanoseconds, convert to seconds
    if v > 1e6:
        return v / 1e9
    return v


def load_results(orig_df: pd.DataFrame, model) -> pd.DataFrame:
    df = orig_df[orig_df['model'] == model].copy()
    # parse metadata into columns if present
    if 'metadata' in df.columns:
        meta = df['metadata'].apply(_safe_parse_metadata)
        meta_df = pd.DataFrame(meta.tolist())
        df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    # coerce numeric columns and standardize durations
    for c in ['prompt_eval_count', 'eval_count', 'latency_sec', 'total_duration', 'prompt_eval_duration', 'eval_duration']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # If latency_sec missing, try to compute from total_duration
    if 'latency_sec' not in df.columns or df['latency_sec'].isna().all():
        if 'total_duration' in df.columns:
            # total_duration may be in ns: convert to seconds with heuristic
            df['latency_sec'] = df['total_duration'].apply(_ns_to_s)

    # Convert durations that look like ns in other fields to seconds as well
    for d in ['prompt_eval_duration', 'eval_duration', 'total_duration']:
        if d in df.columns:
            df[d + '_s'] = df[d].apply(_ns_to_s)

    # ensure token columns exist
    if 'prompt_eval_count' not in df.columns:
        df['prompt_eval_count'] = np.nan
    if 'eval_count' not in df.columns:
        df['eval_count'] = np.nan

    # total tokens
    df['total_tokens'] = df['prompt_eval_count'].fillna(0) + df['eval_count'].fillna(0)

    return df


def compute_binned_metric(df: pd.DataFrame, x_col: str, y_col: str, metric_col: str, x_bins: int = 30, y_bins: int = 30,
                          log_bins: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin x/y into x_bins/y_bins and compute mean(metric) in each 2D bin.

    Returns (x_centers, y_centers, Z) where Z is a 2D array with shape (y_bins, x_bins)
    suitable for plotting with imshow or seaborn. Z contains NaN where no samples in bin.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    metric = df[metric_col].to_numpy(dtype=float)

    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(metric))
    x = x[mask]
    y = y[mask]
    metric = metric[mask]

    if len(x) == 0:
        raise ValueError('No data points for binning')

    # build bin edges
    def edges(vals, bins):
        vals = np.array(vals)
        vmin = vals.min()
        vmax = vals.max()
        if vmin <= 0 or not log_bins:
            return np.linspace(vmin, vmax, bins + 1)
        else:
            # logspace edges
            return np.logspace(math.log10(max(vmin, 1e-6)), math.log10(max(vmax, 1e-6)), bins + 1)

    x_edges = edges(x, x_bins)
    y_edges = edges(y, y_bins)

    # digitize
    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1

    Z = np.full((y_bins, x_bins), np.nan, dtype=float)
    counts = np.zeros((y_bins, x_bins), dtype=int)

    for xi, yi, m in zip(x_idx, y_idx, metric):
        if 0 <= xi < x_bins and 0 <= yi < y_bins:
            if np.isnan(Z[yi, xi]):
                Z[yi, xi] = m
                counts[yi, xi] = 1
            else:
                Z[yi, xi] = (Z[yi, xi] * counts[yi, xi] + m) / (counts[yi, xi] + 1)
                counts[yi, xi] += 1

    # centers for ticks
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    return x_centers, y_centers, Z


def plot_heatmap(x_centers, y_centers, Z, xlabel: str, ylabel: str, title: str, out_path: str, cmap: str = 'viridis'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Z shape is (y, x) where y is rows
    sns.heatmap(Z, xticklabels=[f"{v:.0f}" for v in x_centers], yticklabels=[f"{v:.0f}" for v in y_centers],
                cmap=cmap, ax=ax, cbar_kws={'label': title}, fmt='.2f')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # rotate x ticks
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Latency / Cost heatmap by prompt/output token counts")
    p.add_argument('--csv', default='results_local.csv')
    p.add_argument('--out-dir', default='heatmaps')
    p.add_argument('--metric', choices=['latency', 'eval_duration', 'cost'], default='latency')
    p.add_argument('--cost-per-token', type=float, default=None,
                   help='If metric==cost, multiply total tokens by this value (currency units per token)')
    p.add_argument('--x-bins', type=int, default=40)
    p.add_argument('--y-bins', type=int, default=40)
    p.add_argument('--no-log-bins', dest='log_bins', action='store_false')
    p.set_defaults(log_bins=True)
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)
    total_df = pd.read_csv(args.csv)

    # load results per model
    for model in total_df['model'].unique():
        print(f"Model found: {model}")
        df = load_results(total_df, model)

        # define columns
        x_col = 'prompt_eval_count'
        y_col = 'eval_count'

        if args.metric == 'latency':
            if 'latency_sec' not in df.columns:
                raise ValueError('latency_sec not available in CSV or metadata')
            metric_col = 'latency_sec'
            title = 'Mean Observed Latency (s)'
            cmap = 'rocket'
        elif args.metric == 'eval_duration':
            if 'eval_duration_s' not in df.columns:
                raise ValueError('eval_duration not available in metadata')
            metric_col = 'eval_duration_s'
            title = 'Mean Eval Duration (s)'
            cmap = 'mako'
        else:  # cost
            if args.cost_per_token is None:
                raise ValueError('cost-per-token must be provided when metric==cost')
            df['cost'] = (df['prompt_eval_count'].fillna(0) + df['eval_count'].fillna(0)) * args.cost_per_token
            metric_col = 'cost'
            title = f'Mean Cost (per call) @ {args.cost_per_token:.6f}/token'
            cmap = 'viridis'

        # compute binned metric
        x_centers, y_centers, Z = compute_binned_metric(df, x_col, y_col, metric_col,
                                                        x_bins=args.x_bins, y_bins=args.y_bins, log_bins=args.log_bins)

        # Save heatmap
        os.makedirs(args.out_dir, exist_ok=True)
        out_name = f"{args.metric}_heatmap_{model}.png"
        out_path = os.path.join(args.out_dir, out_name)
        plot_heatmap(x_centers, y_centers, Z, xlabel='Prompt tokens', ylabel='Output tokens', title=title, out_path=out_path, cmap=cmap)
        print(f"Saved heatmap to {out_path}")


if __name__ == '__main__':
    main()
