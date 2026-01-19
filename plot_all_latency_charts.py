import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def load_ollama_results(
    csv_path: str,
    metadata_col: str = "metadata",
    duration_unit: str = "ns"
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    def parse_metadata(meta):
        if pd.isna(meta):
            return {}

        try:
            meta_str = str(meta).strip()
            if not meta_str.startswith("{"):
                meta_str = "{" + meta_str + "}"
            parsed = ast.literal_eval(meta_str)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    # Expand metadata into columns
    metadata_df = df[metadata_col].apply(parse_metadata).apply(pd.Series)

    # Merge and drop original metadata column
    df = pd.concat([df.drop(columns=[metadata_col]), metadata_df], axis=1)

    # Expected numeric columns
    numeric_cols = [
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "prompt_eval_count",
        "eval_count",
        "eval_duration",
    ]

    # Coerce types safely
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert durations to milliseconds
    if duration_unit == "ns":
        duration_cols = [
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ]
        for col in duration_cols:
            if col in df.columns:
                df[col] = df[col] / 1e6  # ms

    # Derived metrics
    if {"prompt_eval_count", "eval_count"}.issubset(df.columns):
        df["total_tokens"] = df["prompt_eval_count"] + df["eval_count"]

    if {"total_duration", "load_duration"}.issubset(df.columns):
        df["actual_total_duration"] = df["total_duration"] - df["load_duration"]

    if {"eval_count", "eval_duration"}.issubset(df.columns):
        df["tokens_per_sec"] = df["eval_count"] / (df["eval_duration"] / 1000)

    return df

df = load_ollama_results("results_local.csv")

MODEL_STYLES = {
    "tinyllama":   {"marker": "o", "color": "tab:blue"},
    "gemma:2b":    {"marker": "s", "color": "tab:green"},
    "qwen2:1.5b":  {"marker": "^", "color": "tab:orange"},
}

def plot_p50_and_regression(x, y, color, bins=20):
    # 1. Clean data: Remove NaNs and zeros (critical for log scale)
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0)
    x, y = x[mask], y[mask]

    # ---------- P50 TREND (Solid Line) ----------
    # Divide data into bins containing equal numbers of points
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, quantiles)

    centers = []
    medians = []

    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        # Only plot the bin if it has at least 5 points (prevents "noisy" lines)
        if m.sum() >= 1:
            centers.append(np.median(x[m]))
            medians.append(np.median(y[m]))

    if len(centers) >= 2:
        plt.plot(centers, medians, color=color, linewidth=3, 
                 linestyle="-", label="P50 Trend", zorder=5)

    # ---------- REGRESSION (Dashed Line) ----------
    # Fits y = m * log10(x) + b
    x_log = np.log10(x)
    coef = np.polyfit(x_log, y, 1)

    # Generate points for a smooth curve on the plot
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    y_fit = coef[0] * np.log10(x_fit) + coef[1]

    plt.plot(x_fit, y_fit, color=color, linewidth=3, 
             linestyle=":", label="Log-Regression", zorder=6)

# Create output directory and helper to save plots
OUTPUT_DIR = "all_scatter_latency_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

plt.figure(figsize=(8, 6))

for model, style in MODEL_STYLES.items():
    d = df[df["model"] == model]

    plt.scatter(
        d["prompt_eval_count"],
        d["prompt_eval_duration"],
        label=model,
        marker=style["marker"],
        color=style["color"],
        alpha=0.35,      # ← lower alpha
        s=40,
        zorder=2         # ← behind lines
    )

    plot_p50_and_regression(
        d["prompt_eval_count"].values,
        d["prompt_eval_duration"].values,
        color=style["color"]
    )

    plt.text(
        d["prompt_eval_count"].median(),
        d["prompt_eval_duration"].median(),
        model,
        color=style["color"],
        fontsize=9
    )

plt.xscale("log")
plt.xlabel("Prompt Token Count")
plt.ylabel("Prompt Eval Latency (ms)")
plt.title("Prompt Eval Latency vs Prompt Token Length")
plt.legend()

save_plot("1_prompt_eval_latency_vs_prompt_tokens.png")

plt.figure(figsize=(8, 6))

for model, style in MODEL_STYLES.items():
    d = df[df["model"] == model]

    plt.scatter(
        d["eval_count"],
        d["eval_duration"],
        label=model,
        marker=style["marker"],
        color=style["color"],
        alpha=0.35,      # ← lower alpha
        s=40,
        zorder=2         # ← behind lines
    )

    plot_p50_and_regression(
        d["eval_count"].values,
        d["eval_duration"].values,
        color=style["color"]
    )

    plt.text(
        d["eval_count"].median(),
        d["eval_duration"].median(),
        model,
        color=style["color"],
        fontsize=9
    )

plt.xscale("log")
plt.xlabel("Output Token Count")
plt.ylabel("Generation Latency (ms)")
plt.title("Eval Latency vs Output Token Length")
plt.legend()

save_plot("2_eval_latency_vs_output_tokens.png")

plt.figure(figsize=(8, 6))

for model, style in MODEL_STYLES.items():
    d = df[df["model"] == model]

    plt.scatter(
        d["total_tokens"],
        d["actual_total_duration"],
        label=model,
        marker=style["marker"],
        color=style["color"],
        alpha=0.35,      # ← lower alpha
        s=40,
        zorder=2         # ← behind lines
    )

    plot_p50_and_regression(
        d["total_tokens"].values,
        d["actual_total_duration"].values,
        color=style["color"]
    )

    plt.text(
        d["total_tokens"].median(),
        d["actual_total_duration"].median(),
        model,
        color=style["color"],
        fontsize=9
    )

plt.xscale("log")
plt.xlabel("Total Tokens (Prompt + Output)")
plt.ylabel("Total Latency (ms) excluding Load Time")
plt.title("Latency vs Total Tokens")
plt.legend()

save_plot("3_latency_vs_total_tokens.png")

plt.figure(figsize=(8, 6))
for model, style in MODEL_STYLES.items():
    d = df[df["model"] == model]
    
    plt.scatter(
        d["eval_count"], 
        d["actual_total_duration"], 
        label=model,
        marker=style["marker"], 
        color=style["color"], 
        alpha=0.35, 
        s=40, 
        zorder=2
    )

    plot_p50_and_regression(
        d["eval_count"].values, 
        d["actual_total_duration"].values, 
        color=style["color"]
    )

    plt.text(d["eval_count"].median(), 
             d["actual_total_duration"].median(), 
             model, 
             color=style["color"], 
             fontsize=9
            )

plt.xscale("log")
plt.xlabel("Output Tokens")
plt.ylabel("Total Latency (ms) excluding Load Time")
plt.title("Latency vs Eval Tokens")
plt.legend()
save_plot("4_output_token_vs_eval_duration.png")


plt.figure(figsize=(8, 6))
for model, style in MODEL_STYLES.items():
    d = df[df["model"] == model]
    
    plt.scatter(d["prompt_eval_count"], d["actual_total_duration"], label=model,
                marker=style["marker"], color=style["color"], alpha=0.35, s=40, zorder=2)

    plot_p50_and_regression(d["prompt_eval_count"].values, d["actual_total_duration"].values, color=style["color"])

    plt.text(d["prompt_eval_count"].median(), d["actual_total_duration"].median(), model, 
             color=style["color"], fontsize=9)

plt.xscale("log")
plt.xlabel("Prompt Token Count")
plt.ylabel("Total Latency (ms) excluding Load Time")
plt.title("Latency vs Prompt Tokens")
plt.legend()
save_plot("5_prompt_token_vs_prompt_duration.png")


