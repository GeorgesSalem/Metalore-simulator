import time
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from make_env import make_env, describe_env
from run_config import MODEL_PATH, RESULTS_DIR, SCENARIO, EVAL_EPISODES, SEED


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if len(series) else None

def print_final_summary(episodes_df, steps_df, jobs_df):
    """Print clean final metrics for the report/presentation."""
    em = episodes_df.mean(numeric_only=True)
    sm = steps_df.mean(numeric_only=True)

    jobs_generated = em.get("jobs_generated")
    jobs_transmitted = em.get("jobs_transmitted")
    jobs_processed = em.get("jobs_processed")
    completion_rate = em.get("job_completion_rate")
    mean_aori = em.get("mean_aori")
    mean_aosi = em.get("mean_aosi")

    pp = sm.get("bw_split", sm.get("action_bw_split"))
    pc = sm.get("comp_split", sm.get("action_comp_split"))

    print("\n=== FINAL SYSTEM METRICS ===")
    print(f"Jobs generated per episode:   {jobs_generated:.0f}")
    print(f"Jobs transmitted per episode: {jobs_transmitted:.0f}")
    print(f"Jobs processed per episode:   {jobs_processed:.0f}")
    print(f"Completion rate:              {completion_rate * 100:.1f}%")
    print(f"Mean AoRI:                    {mean_aori:.2f} steps")
    print(f"Mean AoSI:                    {mean_aosi:.2f} steps")
    print(f"Mean Pp / bandwidth to UEs:   {pp:.3f}")
    print(f"Mean Pc / compute to UEs:     {pc:.3f}")

    if jobs_df is not None and not jobs_df.empty and "entity_type" in jobs_df.columns:
        print("\n=== JOBS BY TYPE PER EPISODE ===")
        jobs_per_type = (jobs_df.groupby("entity_type").size() / len(episodes_df)).round(0).astype(int)
        print(jobs_per_type.to_string())

        print("\n=== IMPORTANT DELAY/FRESHNESS BY TYPE ===")
        cols = [c for c in ["aori", "aosi"] if c in jobs_df.columns]

        if cols:
            summary = jobs_df.groupby("entity_type")[cols].agg(
                ["mean", lambda x: x.quantile(0.95)]
            ).round(2)

            summary = summary.rename(columns={"<lambda_0>": "p95"})

            print(summary.to_string())

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    env = make_env(SCENARIO, seed=SEED)
    model = PPO.load(str(MODEL_PATH), env=env, device="cpu")

    print("Evaluating model:", MODEL_PATH)
    print("Scenario:", SCENARIO)
    print("Environment:", describe_env(env))

    step_rows = []
    episode_rows = []
    all_job_logs = []

    start = time.perf_counter()

    for episode in range(1, EVAL_EPISODES + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            step += 1

            # Metrics recorded by env.metrics.record() inside env.step()
            latest = {k: v[-1] for k, v in env.metrics.step_totals.items() if v and k != "observation"}

            row = {
                "episode": episode,
                "step": step,
                "reward": float(reward),
                "total_reward_so_far": total_reward,
                "obs_ue_queue": float(obs[0]),
                "obs_sensor_queue": float(obs[1]),
                "action_bw_split": float(action[0]),
                "action_comp_split": float(action[1]),
            }
            row.update(latest)
            step_rows.append(row)
            obs = next_obs

        ep_summary = env.metrics.finalize(env.job_tracker)
        job_df = env.job_tracker.to_dataframe()
        if not job_df.empty:
            job_df.insert(0, "episode", episode)
            all_job_logs.append(job_df)

        episode_rows.append({
            "episode": episode,
            "steps": step,
            "total_reward": total_reward,
            "jobs_generated": ep_summary.get("jobs_generated"),
            "jobs_transmitted": ep_summary.get("jobs_transmitted"),
            "jobs_processed": ep_summary.get("jobs_processed"),
            "job_completion_rate": ep_summary.get("job_completion_rate"),
            "mean_aoi": ep_summary.get("mean_aoi"),
            "mean_aori": ep_summary.get("mean_aori"),
            "mean_aosi": ep_summary.get("mean_aosi"),
        })
        print(f"Episode {episode}: reward={total_reward:.3f}, steps={step}, processed={ep_summary.get('jobs_processed')}")

    elapsed = time.perf_counter() - start

    steps_df = pd.DataFrame(step_rows)
    episodes_df = pd.DataFrame(episode_rows)

    steps_path = RESULTS_DIR / "evaluation_steps.csv"
    episodes_path = RESULTS_DIR / "evaluation_episodes.csv"
    jobs_path = RESULTS_DIR / "evaluation_jobs.csv"

    steps_df.to_csv(steps_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)

    jobs_df = pd.DataFrame()
    if all_job_logs:
        jobs_df = pd.concat(all_job_logs, ignore_index=True)
        jobs_df.to_csv(jobs_path, index=False)

    print("\nEvaluation finished.")
    print("Total time:", round(elapsed, 2), "seconds")
    print("Average episode reward:", round(float(episodes_df["total_reward"].mean()), 3))
    print("Mean AoRI:", safe_mean(episodes_df["mean_aori"]))
    print("Mean AoSI:", safe_mean(episodes_df["mean_aosi"]))
    print("Saved:", steps_path)
    print("Saved:", episodes_path)
    if all_job_logs:
        print("Saved:", jobs_path)

    print_final_summary(episodes_df, steps_df, jobs_df)

    env.close()


if __name__ == "__main__":
    main()
