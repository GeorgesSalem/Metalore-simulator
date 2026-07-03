from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CHANGE ONLY THESE PATHS
# ============================================================

EXPERIMENTS = {
    "S1_100k": Path(
        r"results\dynamic_sweeps\S1_100k\s1_eval_until_80ue_fixed_25s\balanced_model_50ue_25s\summary.csv"
    ),

    "S1_300k": Path(
        r"results\dynamic_sweeps\S1_300k\s1_eval_until_80ues_fixed_25sensors_300k_PPOHyperparameter\balanced_model_50ue_25s\summary.csv"
    ),

    "S2_300k": Path(
        r"results\dynamic_sweeps\S2_300k\s2_eval_until_80ues_fixed_25sensors_300k\balanced_model_60ue_35s\summary.csv"
    ),

    "S2_300k_RP": Path(
        r"results\dynamic_sweeps\S2_300k_RP\s2_until_80ues_fixed_25sensors_300k\balanced_high_load_model_60ue_35s\summary.csv"
    ),
}

OUTPUT_DIR = Path(
    r"results\benchmark_comparisons\S1_S2_S3_S4_comparison"
)

# ============================================================


KEY_COLS = ["num_ues", "num_sensors", "total_devices"]

METRICS = {
    "job_completion_rate": {
        "label": "Job completion rate",
        "higher_is_better": True,
        "file": "completion_rate_comparison.png",
    },
    "mean_aori": {
        "label": "Mean AoRI (timesteps)",
        "higher_is_better": False,
        "file": "aori_comparison.png",
    },
    "mean_aosi": {
        "label": "Mean AoSI (timesteps)",
        "higher_is_better": False,
        "file": "aosi_comparison.png",
    },
    "total_reward": {
        "label": "Total reward",
        "higher_is_better": True,
        "file": "reward_comparison.png",
    },
}


def load_summary(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found: {path}")

    df = pd.read_csv(path)

    required_cols = KEY_COLS + list(METRICS.keys())
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    df = df[required_cols].copy()

    rename_cols = {
        metric: f"{metric}__{name}"
        for metric in METRICS.keys()
    }

    return df.rename(columns=rename_cols)


def merge_all_experiments() -> pd.DataFrame:
    merged = None

    for name, path in EXPERIMENTS.items():
        df = load_summary(name, path)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=KEY_COLS, how="inner")

    if merged is None or merged.empty:
        raise ValueError(
            "No matching rows found between the 4 CSV files. "
            "Make sure all 4 CSVs were evaluated on the same sweep points "
            "with the same num_ues, num_sensors, and total_devices."
        )

    return merged.sort_values(KEY_COLS).reset_index(drop=True)


def make_x_labels(df: pd.DataFrame):
    return [
        f"{int(row.num_ues)} UE\n{int(row.num_sensors)} S\nT={int(row.total_devices)}"
        for row in df.itertuples()
    ]


def plot_metric(df: pd.DataFrame, metric: str, info: dict) -> None:
    labels = make_x_labels(df)
    x = range(len(df))

    plt.figure(figsize=(14, 6))

    for name in EXPERIMENTS.keys():
        col = f"{metric}__{name}"
        plt.plot(x, df[col], marker="o", label=name)

        for i, value in enumerate(df[col]):
            plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    plt.xticks(x, labels)
    plt.ylabel(info["label"])
    plt.title(f"Comparison of 4 models - {info['label']}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / info["file"], dpi=300)
    plt.close()


def build_ranking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name in EXPERIMENTS.keys():
        row = {"model": name}

        row["avg_completion"] = df[f"job_completion_rate__{name}"].mean()
        row["avg_aori"] = df[f"mean_aori__{name}"].mean()
        row["avg_aosi"] = df[f"mean_aosi__{name}"].mean()
        row["avg_reward"] = df[f"total_reward__{name}"].mean()

        rows.append(row)

    ranking = pd.DataFrame(rows)

    ranking["rank_completion"] = ranking["avg_completion"].rank(
        ascending=False, method="min"
    )
    ranking["rank_aori"] = ranking["avg_aori"].rank(
        ascending=True, method="min"
    )
    ranking["rank_aosi"] = ranking["avg_aosi"].rank(
        ascending=True, method="min"
    )
    ranking["rank_reward"] = ranking["avg_reward"].rank(
        ascending=False, method="min"
    )

    ranking["total_rank"] = (
        ranking["rank_completion"]
        + ranking["rank_aori"]
        + ranking["rank_aosi"]
        + ranking["rank_reward"]
    )

    ranking = ranking.sort_values("total_rank").reset_index(drop=True)

    return ranking


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = merge_all_experiments()

    comparison_path = OUTPUT_DIR / "comparison_4_models.csv"
    comparison.to_csv(comparison_path, index=False)

    for metric, info in METRICS.items():
        plot_metric(comparison, metric, info)

    ranking = build_ranking(comparison)

    ranking_path = OUTPUT_DIR / "ranking_4_models.csv"
    ranking.to_csv(ranking_path, index=False)

    print("\nSaved comparison CSV:")
    print(comparison_path)

    print("\nSaved ranking CSV:")
    print(ranking_path)

    print("\nSaved plots in:")
    print(OUTPUT_DIR)

    print("\nRanking:")
    print(ranking.to_string(index=False))

    print("\nBest model:")
    print(ranking.iloc[0]["model"])


if __name__ == "__main__":
    main()