import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _safe_literal_eval(val):
    """Safely parse the metadata column which may be a dict string (single quotes) or JSON."""
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return pd.json.loads(val)
        except Exception:
            return {}


def _ns_to_s(x):
    """Convert nanoseconds-ish numbers to seconds if they look large; otherwise return NaN for missing."""
    if x is None:
        return np.nan
    try:
        x = float(x)
    except Exception:
        return np.nan
    # If value seems in nanoseconds (> 1e6), convert to seconds
    if x > 1e6:
        return x / 1e9
    return x


def create_bar_chart(csv_path='results_local.csv', out_prefix='avg_by_model'):
    """
    Create charts and a summary CSV that include all available model information found in the results CSV.

    Outputs:
      - {out_prefix}.png : bar chart of average score by model
      - {out_prefix}_durations.png : grouped bar chart showing latency and major durations
      - model_summary.csv : per-model aggregated metrics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if 'model' not in df.columns:
        raise ValueError("CSV must contain a 'model' column")

    # Ensure numeric score/latency
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
    if 'latency_sec' in df.columns:
        df['latency_sec'] = pd.to_numeric(df['latency_sec'], errors='coerce')

    # Parse metadata column into separate columns if present
    meta_cols = ['total_duration', 'load_duration', 'prompt_eval_duration', 'prompt_eval_count', 'eval_count', 'eval_duration']
    if 'metadata' in df.columns:
        parsed = df['metadata'].apply(_safe_literal_eval)
        for col in meta_cols:
            df[col] = parsed.apply(lambda m: m.get(col) if isinstance(m, dict) else np.nan)

    # Convert durations that appear to be in nanoseconds to seconds where appropriate
    for dcol in ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']:
        if dcol in df.columns:
            df[dcol + '_s'] = df[dcol].apply(_ns_to_s)

    # Aggregate per model
    agg_funcs = {
        'score': 'mean',
        'latency_sec': 'mean'
    }
    # include counts and metadata aggregations
    for c in ['total_duration_s', 'load_duration_s', 'prompt_eval_duration_s', 'prompt_eval_count', 'eval_count', 'eval_duration_s']:
        if c in df.columns:
            agg_funcs[c] = 'mean'

    summary = df.groupby('model', observed=True).agg(agg_funcs)
    summary = summary.rename(columns={
        'score': 'avg_score',
        'latency_sec': 'avg_latency_sec'
    })
    summary['samples'] = df.groupby('model', observed=True).size()

    # Save a summary CSV for downstream inspection
    summary_csv = 'model_summary.csv'
    summary.to_csv(summary_csv)
    print(f"Saved model summary to {summary_csv}")

    # Create score bar chart
    png_score = f"{out_prefix}_score.png"
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary.index, summary['avg_score'].fillna(0), color='C0')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)
    plt.title('Average Score by Model')
    for bar, val, cnt in zip(bars, summary['avg_score'], summary['samples']):
        if not np.isnan(val):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}\n(n={int(cnt)})", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(png_score, dpi=150)
    print(f"Saved bar chart to {png_score}")

    # Create duration/latency grouped chart if we have any duration columns
    dur_cols = [c for c in ['avg_latency_sec', 'total_duration_s', 'prompt_eval_duration_s', 'eval_duration_s'] if c in summary.columns]
    if len(dur_cols) > 0:
        png_dur = f"{out_prefix}_durations.png"
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(summary.index))
        width = 0.2
        offsets = np.linspace(-width, width, len(dur_cols))
        for i, col in enumerate(dur_cols):
            vals = summary[col].fillna(0).values
            ax.bar(x + offsets[i], vals, width=width/len(dur_cols), label=col)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index)
        ax.set_ylabel('Latency (seconds)')
        ax.set_title('Latency and Duration metrics by model (seconds)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(png_dur, dpi=150)
        print(f"Saved duration chart to {png_dur}")

    return summary

if __name__ == "__main__":
    create_bar_chart()