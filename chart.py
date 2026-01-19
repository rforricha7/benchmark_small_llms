import os
import pandas as pd
import matplotlib.pyplot as plt

def create_bar_chart(csv_path='results_local.csv', out_path='avg_score_by_model.png'):
    """
    Create a bar chart of average score by model from results CSV and save to out_path.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if 'model' not in df.columns or 'score' not in df.columns:
        raise ValueError("CSV must contain 'model' and 'score' columns")

    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    agg = df.groupby('model', observed=True)['score'].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(agg.index, agg.values, color='C0')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)
    plt.title('Average Score by Model')
    for bar, val in zip(bars, agg.values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved bar chart to {out_path}")

if __name__ == "__main__":
    create_bar_chart()