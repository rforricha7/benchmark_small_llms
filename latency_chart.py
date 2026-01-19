import os
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def _safe_parse_metadata(s):
    if pd.isna(s):
        return {}
    try:
        # metadata in CSV is a Python dict string using single quotes
        return ast.literal_eval(s)
    except Exception:
        # fallback: try JSON-like replacement
        try:
            return ast.literal_eval(s.replace("None", "null").replace("'", '"'))
        except Exception:
            return {}

def _median_trendline(ax, x, y, bins=20, color="C3", label="p50 (binned)"):
    # compute median y per x-bin and plot
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return
    df["bin"] = pd.qcut(df["x"].rank(method="first"), q=min(bins, max(1, len(df))), duplicates="drop")
    med = df.groupby("bin").agg(x=("x", "median"), y=("y", "median")).sort_values("x")
    ax.plot(med["x"], med["y"], marker="o", linestyle="-", color=color, label=label)

def _regression_line(ax, x, y, color="C1", label="linear fit"):
    # simple linear regression (numpy polyfit)
    mask = (~np.isnan(x)) & (~np.isnan(y)) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    coeffs = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
    ys = np.polyval(coeffs, xs)
    ax.plot(xs, ys, color=color, linestyle="--", label=label)

def create_latency_scatter(csv_path="results_local.csv", out_dir="plots"):
    """
    Create scatter plots diagnostic for latency vs token counts.

    Produces:
      - prompt_eval_count vs prompt_eval_duration
      - eval_count vs eval_duration
      - total_tokens (prompt_eval_count + eval_count) vs latency_sec

    Each plot contains:
      - scatter points colored by model
      - linear regression line (dashed, overall)
      - p50 trendline computed by binning (solid, overall)
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    # parse metadata column into separate numeric columns
    meta_parsed = df["metadata"].apply(_safe_parse_metadata)
    df_meta = pd.DataFrame(meta_parsed.tolist()).fillna(value=np.nan)
    # normalize column names we care about
    for col in ["prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]:
        if col not in df_meta.columns:
            df_meta[col] = np.nan

    df = pd.concat([df.reset_index(drop=True), df_meta[["prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]].reset_index(drop=True)], axis=1)

    # numeric conversions
    df["prompt_eval_count"] = pd.to_numeric(df["prompt_eval_count"], errors="coerce")
    df["prompt_eval_duration"] = pd.to_numeric(df["prompt_eval_duration"], errors="coerce")
    df["eval_count"] = pd.to_numeric(df["eval_count"], errors="coerce")
    df["eval_duration"] = pd.to_numeric(df["eval_duration"], errors="coerce")
    df["latency_sec"] = pd.to_numeric(df["latency_sec"], errors="coerce")

    # total tokens (fallbacks if one of counts missing)
    df["total_tokens"] = df["prompt_eval_count"].fillna(0) + df["eval_count"].fillna(0)

    # prepare model color mapping
    models = df["model"].dropna().unique().tolist()
    palette = {}
    if models:
        colors = sns.color_palette("tab10", n_colors=max(3, len(models)))
        palette = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    plots = [
        {
            "x": "prompt_eval_count",
            "y": "prompt_eval_duration",
            "xlabel": "Prompt token count (prompt_eval_count)",
            "ylabel": "Prompt eval duration (prompt_eval_duration)",
            "fname": "prompt_tokens_vs_prompt_duration.png",
            "y_scale": None
        },
        {
            "x": "eval_count",
            "y": "eval_duration",
            "xlabel": "Output token count (eval_count)",
            "ylabel": "Eval duration (eval_duration)",
            "fname": "output_tokens_vs_eval_duration.png",
            "y_scale": None
        },
        {
            "x": "total_tokens",
            "y": "latency_sec",
            "xlabel": "Total tokens (prompt + output)",
            "ylabel": "Observed latency (seconds)",
            "fname": "total_tokens_vs_latency.png",
            "y_scale": None
        },
        # optional convenience: prompt tokens vs observed latency
        {
            "x": "prompt_eval_count",
            "y": "latency_sec",
            "xlabel": "Prompt token count (prompt_eval_count)",
            "ylabel": "Observed latency (seconds)",
            "fname": "prompt_tokens_vs_latency.png",
            "y_scale": None
        },
        # optional: output tokens vs observed latency
        {
            "x": "eval_count",
            "y": "latency_sec",
            "xlabel": "Output token count (eval_count)",
            "ylabel": "Observed latency (seconds)",
            "fname": "output_tokens_vs_latency.png",
            "y_scale": None
        },
    ]

    for p in plots:
        x_all = df[p["x"]].to_numpy(dtype=float)
        y_all = df[p["y"]].to_numpy(dtype=float)

        # drop extreme empty
        if np.all(np.isnan(x_all)) or np.all(np.isnan(y_all)):
            # skip empty plot
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # plot each model separately so colors/legend show per-model
        for m in models:
            mask = df["model"] == m
            x_m = df.loc[mask, p["x"]].to_numpy(dtype=float)
            y_m = df.loc[mask, p["y"]].to_numpy(dtype=float)
            if x_m.size == 0 or y_m.size == 0 or np.all(np.isnan(x_m)) or np.all(np.isnan(y_m)):
                continue
            ax.scatter(x_m, y_m, alpha=0.8, s=50, color=palette.get(m), label=m, edgecolor="k", linewidth=0.3)

        # add regression line and p50 trendline computed on all points
        try:
            _regression_line(ax, x_all, y_all, color="C1", label="linear fit (all)")
            _median_trendline(ax, x_all, y_all, bins=15, color="C3", label="p50 (binned, all)")
        except Exception:
            pass

        ax.set_xlabel(p["xlabel"])
        ax.set_ylabel(p["ylabel"])
        ax.set_title(f"{p['ylabel']} vs {p['xlabel']}")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        # set log scale for x if distribution is very skewed (heuristic)
        try:
            if np.nanmax(x_all) / (np.nanmin(x_all[np.nonzero(x_all)]) + 1e-12) > 100:
                ax.set_xscale("log")
        except Exception:
            pass
        # set log scale for y if skewed
        try:
            if np.nanmax(y_all) / (np.nanmin(y_all[np.nonzero(y_all)]) + 1e-12) > 100:
                ax.set_yscale("log")
        except Exception:
            pass

        out_path = os.path.join(out_dir, p["fname"])
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    create_latency_scatter()