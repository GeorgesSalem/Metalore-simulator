from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from run_config import RESULTS_DIR


def plot_csv_column(csv_path: Path, x_col: str, y_col: str, out_name: str):
    if not csv_path.exists():
        print(f"Missing {csv_path}, skip {y_col}")
        return
    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Column missing in {csv_path}: {x_col} or {y_col}")
        return
    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(y_col)
    out_path = RESULTS_DIR / out_name
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_csv_column(RESULTS_DIR / "evaluation_episodes.csv", "episode", "total_reward", "eval_episode_reward.png")
    plot_csv_column(RESULTS_DIR / "evaluation_episodes.csv", "episode", "mean_aori", "eval_mean_aori.png")
    plot_csv_column(RESULTS_DIR / "evaluation_episodes.csv", "episode", "mean_aosi", "eval_mean_aosi.png")
    plot_csv_column(RESULTS_DIR / "evaluation_steps.csv", "step", "reward", "eval_step_reward.png")


if __name__ == "__main__":
    main()
