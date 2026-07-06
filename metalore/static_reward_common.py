from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Path setup: place these scripts inside the metalore/ folder.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if (ROOT / "make_env.py").exists():
    METALORE_DIR = ROOT
else:
    METALORE_DIR = ROOT.parent

PROJECT_ROOT = METALORE_DIR.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(METALORE_DIR))

from make_env import describe_env, make_env_from_config
from metalore.config import default_config
from metalore.scenarios import SingleCellEnv


# ============================================================
# GLOBAL STATIC SCENARIO SETTINGS
# ============================================================

UE_VALUES = [20, 40, 60]
SENSOR_VALUES = [10, 20]
NUM_SENSORS = SENSOR_VALUES[0]
MAX_STEPS = 100

# PPO hyperparameters are fixed in all experiments.
PPO_PARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.20,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_epochs=10,
    device="cpu",
)

# Fixed reward parameters. We only tune eta and c2.
SYNC_BASE_REWARD = 10.0
E2E_DELAY_THRESHOLD = 2.0

# Fixed BS setup.
BS_POSITIONS = [(100.0, 100.0)]
BS_COMPUTE_CAPACITY = 800
BS_BANDWIDTH = 600e6

# Search/evaluation settings.
TRAIN_TIMESTEPS =100_000      # start small; final selected runs can be changed to 300_000
EVAL_EPISODES = 20            # final selected runs can be changed to 50
SEED = 5555
CHECKPOINT_FREQ = 10_000
SKIP_IF_MODEL_EXISTS = True
SAVE_EVAL_JOB_CSV = False
FAIRNESS_EPS = 1e-6
TARGET_COMPLETION = 0.99

# Default intention used when ranking grid/LLM candidates.
# Options: "user_priority", "sensor_priority", "balanced", "fairness"
OPERATOR_INTENTION = "user_priority"


# ============================================================
# Helpers
# ============================================================

def tag_float(x: float) -> str:
    """Make file-safe tags: -2.5 -> m2p5, 0.7 -> 0p7."""
    return str(x).replace("-", "m").replace(".", "p")


def sensor_tag(num_sensors: int | None = None, sensor_values: List[int] | None = None) -> str:
    """Create a readable sensor-count tag for experiment folders."""
    if num_sensors is not None:
        return str(int(num_sensors))
    if sensor_values is not None:
        return "_".join(str(int(value)) for value in sensor_values)
    return str(int(NUM_SENSORS))


def build_experiment_name(
    prefix: str,
    eta: float | None = None,
    c2: float | None = None,
    num_sensors: int | None = None,
    sensor_values: List[int] | None = None,
) -> str:
    """Create a readable experiment folder name."""
    sensors = sensor_tag(num_sensors=num_sensors, sensor_values=sensor_values)
    if eta is None or c2 is None:
        return f"{prefix}_static_{sensors}s_{OPERATOR_INTENTION}_{TRAIN_TIMESTEPS // 1000}k"
    return (
        f"{prefix}_static_{sensors}s_eta{tag_float(eta)}_c2{tag_float(c2)}_"
        f"{OPERATOR_INTENTION}_{TRAIN_TIMESTEPS // 1000}k"
    )


def build_static_config(
    num_ues: int,
    eta: float,
    c2: float,
    seed: int,
    num_sensors: int | None = None,
) -> Dict:
    """Create one static scenario: fixed UEs, fixed sensors, fixed reward eta/c2."""
    if num_sensors is None:
        num_sensors = NUM_SENSORS

    config = default_config()

    config["environment"]["num_ues"] = int(num_ues)
    config["environment"]["num_sensors"] = int(num_sensors)
    config["environment"]["max_steps"] = int(MAX_STEPS)
    config["environment"]["seed"] = int(seed)

    config["bs"]["positions"] = list(BS_POSITIONS)
    config["bs"]["compute_capacity"] = BS_COMPUTE_CAPACITY
    config["bs"]["bandwidth"] = BS_BANDWIDTH

    # eta = synchronization discount factor
    # c2  = delay penalty
    config["reward"].update({
        "delay_penalty": float(c2),
        "sync_base_reward": float(SYNC_BASE_REWARD),
        "discount_factor": float(eta),
        "e2e_delay_threshold": float(E2E_DELAY_THRESHOLD),
    })

    return config


def compute_episode_fairness(env, num_ues: int) -> Dict[str, float]:
    """
    Fairness from per-UE service rate:
    r_i = processed UE jobs for UE i / generated UE jobs for UE i
    fairness_log_sum = sum_i log(r_i + eps)

    Higher fairness_log_sum is better. Near 0 means all UEs have service rate near 1.
    """
    if not hasattr(env, "job_tracker") or not hasattr(env.job_tracker, "ep_per_entity"):
        return {
            "fairness_log_sum": math.nan,
            "min_ue_service_rate": math.nan,
            "mean_ue_service_rate": math.nan,
        }

    rates: List[float] = []
    for ue_id in range(int(num_ues)):
        counts = env.job_tracker.ep_per_entity.get(("UE", ue_id))
        if counts is None or getattr(counts, "jobs_generated", 0) == 0:
            rate = 1.0
        else:
            rate = getattr(counts, "jobs_processed", 0) / getattr(counts, "jobs_generated", 1)
        rates.append(max(float(rate), FAIRNESS_EPS))

    return {
        "fairness_log_sum": sum(math.log(r) for r in rates),
        "min_ue_service_rate": min(rates) if rates else math.nan,
        "mean_ue_service_rate": sum(rates) / len(rates) if rates else math.nan,
    }


def score_result(row: Dict, operator_intention: str = OPERATOR_INTENTION) -> float:
    """Lower score is better. Used to select best eta/c2 for each UE number."""
    completion = float(row.get("job_completion_rate", 0.0) or 0.0)
    completion_gap = max(0.0, TARGET_COMPLETION - completion) * 100.0
    aori = float(row.get("mean_aori", 0.0) or 0.0)
    aosi = float(row.get("mean_aosi", 0.0) or 0.0)
    fairness = float(row.get("fairness_log_sum", 0.0) or 0.0)

    if operator_intention == "user_priority":
        return 10.0 * completion_gap + 3.0 * aori + 1.0 * aosi
    if operator_intention == "sensor_priority":
        return 10.0 * completion_gap + 1.0 * aori + 3.0 * aosi
    if operator_intention == "balanced":
        return 10.0 * completion_gap + 2.0 * aori + 2.0 * aosi
    if operator_intention == "fairness":
        return 10.0 * completion_gap + 0.5 * aori + 0.5 * aosi - fairness

    raise ValueError(f"Unknown operator_intention={operator_intention}")


def estimate_reward_convergence(run_dir: Path, train_time_seconds: float) -> Dict[str, float]:
    """
    Approximate convergence using SB3 Monitor rewards.
    It returns the first timestep where rolling episode reward reaches 95% of final rolling reward.
    """
    files = list(run_dir.glob("*monitor*.csv"))
    if not files:
        return {
            "reward_convergence_episode": math.nan,
            "reward_convergence_timestep": math.nan,
            "reward_convergence_time_est_seconds": math.nan,
        }

    try:
        df = pd.read_csv(files[0], comment="#")
    except Exception:
        return {
            "reward_convergence_episode": math.nan,
            "reward_convergence_timestep": math.nan,
            "reward_convergence_time_est_seconds": math.nan,
        }

    if df.empty or "r" not in df.columns or "l" not in df.columns:
        return {
            "reward_convergence_episode": math.nan,
            "reward_convergence_timestep": math.nan,
            "reward_convergence_time_est_seconds": math.nan,
        }

    window = min(10, len(df))
    roll = df["r"].rolling(window=window, min_periods=1).mean()
    final_value = float(roll.iloc[-1])
    threshold = 0.95 * final_value if final_value > 0 else final_value
    mask = roll >= threshold

    if not mask.any():
        return {
            "reward_convergence_episode": math.nan,
            "reward_convergence_timestep": math.nan,
            "reward_convergence_time_est_seconds": math.nan,
        }

    idx = int(mask.idxmax())
    conv_episode = idx + 1
    conv_timestep = int(df["l"].iloc[: idx + 1].sum())
    conv_time_est = train_time_seconds * conv_timestep / TRAIN_TIMESTEPS if TRAIN_TIMESTEPS else math.nan

    return {
        "reward_convergence_episode": conv_episode,
        "reward_convergence_timestep": conv_timestep,
        "reward_convergence_time_est_seconds": conv_time_est,
    }


def train_one_model(
    *,
    method_name: str,
    experiment_name: str,
    num_ues: int,
    num_sensors: int | None = None,
    eta: float,
    c2: float,
    model_path: Path,
    run_dir: Path,
) -> Dict:
    """Train one PPO model for one static scenario and one reward pair."""
    if num_sensors is None:
        num_sensors = NUM_SENSORS

    config = build_static_config(
        num_ues=num_ues,
        num_sensors=num_sensors,
        eta=eta,
        c2=c2,
        seed=SEED,
    )

    if model_path.exists() and SKIP_IF_MODEL_EXISTS:
        print("Model already exists, skipping training:", model_path)
        return {
            "training_time_seconds": math.nan,
            "steps_per_second": math.nan,
            "trained": False,
            "reward_convergence_episode": math.nan,
            "reward_convergence_timestep": math.nan,
            "reward_convergence_time_est_seconds": math.nan,
        }

    env = make_env_from_config(config, env_cls=SingleCellEnv, seed=SEED)
    env = Monitor(env, filename=str(run_dir / "monitor_train"))

    print("\nTraining PPO")
    print("Method:", method_name)
    print("Experiment:", experiment_name)
    print("UE / sensors:", num_ues, "/", num_sensors)
    print("eta:", eta, "c2:", c2)
    print("Environment:", describe_env(env.unwrapped if hasattr(env, "unwrapped") else env))
    print("Model path:", model_path)

    model = PPO(
        "MlpPolicy",
        env,
        **PPO_PARAMS,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix=(
            f"checkpoint_{method_name}_{num_ues}ue_{num_sensors}s_"
            f"eta{tag_float(eta)}_c2{tag_float(c2)}"
        ),
    )

    start = time.perf_counter()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    model.save(str(model_path))
    env.close()

    convergence_info = estimate_reward_convergence(run_dir, elapsed)

    return {
        "training_time_seconds": elapsed,
        "steps_per_second": TRAIN_TIMESTEPS / elapsed if elapsed > 0 else math.nan,
        "trained": True,
        **convergence_info,
    }


def evaluate_one_model(
    *,
    method_name: str,
    experiment_name: str,
    num_ues: int,
    num_sensors: int | None = None,
    eta: float,
    c2: float,
    model_path: Path,
    run_dir: Path,
) -> Dict:
    """Evaluate one PPO model on the SAME static scenario used for training."""
    if num_sensors is None:
        num_sensors = NUM_SENSORS

    config = build_static_config(
        num_ues=num_ues,
        num_sensors=num_sensors,
        eta=eta,
        c2=c2,
        seed=SEED,
    )
    env = make_env_from_config(config, env_cls=SingleCellEnv, seed=SEED)
    model = PPO.load(str(model_path), device="cpu")

    episode_rows = []
    step_rows = []
    job_rows = []

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

            latest = {
                k: v[-1]
                for k, v in env.metrics.step_totals.items()
                if v and k != "observation"
            }
            step_row = {
                "method": method_name,
                "experiment": experiment_name,
                "num_ues": num_ues,
                "num_sensors": num_sensors,
                "eta": eta,
                "c2": c2,
                "episode": episode,
                "step": step,
                "reward": float(reward),
                "action_bw_split": float(action[0]),
                "action_comp_split": float(action[1]),
            }
            step_row.update(latest)
            step_rows.append(step_row)
            obs = next_obs

        ep_summary = env.metrics.finalize(env.job_tracker)
        fairness = compute_episode_fairness(env, num_ues=num_ues)

        episode_rows.append({
            "method": method_name,
            "experiment": experiment_name,
            "num_ues": num_ues,
            "num_sensors": num_sensors,
            "eta": eta,
            "c2": c2,
            "episode": episode,
            "steps": step,
            "total_reward": total_reward,
            "jobs_generated": ep_summary.get("jobs_generated"),
            "jobs_transmitted": ep_summary.get("jobs_transmitted"),
            "jobs_processed": ep_summary.get("jobs_processed"),
            "bits_transmitted": ep_summary.get("bits_transmitted"),
            "cycles_processed": ep_summary.get("cycles_processed"),
            "job_completion_rate": ep_summary.get("job_completion_rate"),
            "mean_aori": ep_summary.get("mean_aori"),
            "mean_aosi": ep_summary.get("mean_aosi"),
            **fairness,
        })

        if SAVE_EVAL_JOB_CSV:
            jobs = env.job_tracker.to_dataframe()
            if not jobs.empty:
                jobs.insert(0, "episode", episode)
                jobs.insert(0, "c2", c2)
                jobs.insert(0, "eta", eta)
                jobs.insert(0, "num_sensors", num_sensors)
                jobs.insert(0, "num_ues", num_ues)
                jobs.insert(0, "experiment", experiment_name)
                jobs.insert(0, "method", method_name)
                job_rows.append(jobs)

    env.close()

    episodes_df = pd.DataFrame(episode_rows)
    steps_df = pd.DataFrame(step_rows)
    episodes_df.to_csv(run_dir / "eval_episodes.csv", index=False)
    steps_df.to_csv(run_dir / "eval_steps.csv", index=False)

    if SAVE_EVAL_JOB_CSV and job_rows:
        pd.concat(job_rows, ignore_index=True).to_csv(run_dir / "eval_jobs.csv", index=False)

    summary = episodes_df.agg({
        "total_reward": "mean",
        "jobs_generated": "mean",
        "jobs_transmitted": "mean",
        "jobs_processed": "mean",
        "bits_transmitted": "mean",
        "cycles_processed": "mean",
        "job_completion_rate": "mean",
        "mean_aori": "mean",
        "mean_aosi": "mean",
        "fairness_log_sum": "mean",
        "min_ue_service_rate": "mean",
        "mean_ue_service_rate": "mean",
        "steps": "mean",
    }).to_dict()

    if not steps_df.empty:
        summary.update(steps_df.agg({
            "action_bw_split": "mean",
            "action_comp_split": "mean",
        }).to_dict())

    summary["throughput_jobs_per_step"] = (
        summary["jobs_processed"] / summary["steps"] if summary.get("steps") else math.nan
    )
    summary["throughput_bits_per_step"] = (
        summary["bits_transmitted"] / summary["steps"] if summary.get("steps") else math.nan
    )

    return summary


def run_training_evaluation_for_pairs(
    *,
    method_name: str,
    experiment_name: str,
    pairs_by_ue: Dict[int, List[Tuple[float, float]]] | None = None,
    pairs_by_scenario: Dict[Tuple[int, int], List[Tuple[float, float]]] | None = None,
    operator_intention: str = OPERATOR_INTENTION,
) -> pd.DataFrame:
    """
    Train/evaluate multiple eta/c2 pairs.
    pairs_by_ue example:
    {10: [(0.7, -2.0)], 20: [(0.7, -2.0)]}
    pairs_by_scenario example:
    {(20, 10): [(0.7, -2.0)], (20, 20): [(0.7, -2.0)]}
    """
    if pairs_by_scenario is None:
        if pairs_by_ue is None:
            raise ValueError("Provide pairs_by_ue or pairs_by_scenario.")
        pairs_by_scenario = {
            (int(num_ues), int(NUM_SENSORS)): pairs
            for num_ues, pairs in pairs_by_ue.items()
        }

    scenario_items = sorted(pairs_by_scenario.items())
    ue_values = sorted({int(num_ues) for (num_ues, _), _ in scenario_items})
    sensor_values = sorted({int(num_sensors) for (_, num_sensors), _ in scenario_items})
    static_folder = f"static_{sensor_tag(sensor_values=sensor_values)}s"

    base_dir = METALORE_DIR / "results" / "reward_pipeline" / static_folder / experiment_name
    models_dir = METALORE_DIR / "models" / "reward_pipeline" / static_folder / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    print("\n=== STATIC PPO REWARD EXPERIMENT ===")
    print("Method:", method_name)
    print("Experiment:", experiment_name)
    print("Operator intention:", operator_intention)
    print("UE values:", ue_values)
    print("Sensor values:", sensor_values)
    print("Train timesteps:", TRAIN_TIMESTEPS)
    print("Eval episodes:", EVAL_EPISODES)
    print("Output:", base_dir)

    for (num_ues, num_sensors), pairs in scenario_items:
        for eta, c2 in pairs:
            run_name = (
                f"{num_ues}ue_{num_sensors}s_eta{tag_float(eta)}_"
                f"c2{tag_float(c2)}_{TRAIN_TIMESTEPS // 1000}k"
            )
            run_dir = base_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"ppo_{method_name}_{run_name}.zip"

            train_info = train_one_model(
                method_name=method_name,
                experiment_name=experiment_name,
                num_ues=num_ues,
                num_sensors=num_sensors,
                eta=eta,
                c2=c2,
                model_path=model_path,
                run_dir=run_dir,
            )

            eval_info = evaluate_one_model(
                method_name=method_name,
                experiment_name=experiment_name,
                num_ues=num_ues,
                num_sensors=num_sensors,
                eta=eta,
                c2=c2,
                model_path=model_path,
                run_dir=run_dir,
            )

            row = {
                "method": method_name,
                "experiment": experiment_name,
                "operator_intention": operator_intention,
                "num_ues": num_ues,
                "num_sensors": num_sensors,
                "eta": eta,
                "c2": c2,
                "sync_base_reward": SYNC_BASE_REWARD,
                "e2e_delay_threshold": E2E_DELAY_THRESHOLD,
                "train_timesteps": TRAIN_TIMESTEPS,
                "eval_episodes": EVAL_EPISODES,
                "seed": SEED,
                "model_path": str(model_path),
                "run_dir": str(run_dir),
                **train_info,
                **eval_info,
            }
            row["score"] = score_result(row, operator_intention=operator_intention)
            all_rows.append(row)

            df = pd.DataFrame(all_rows)
            df.to_csv(base_dir / f"{method_name}_all_results.csv", index=False)
            best_by_ue_sensor_df = (
                df.sort_values("score")
                .groupby(["num_ues", "num_sensors"], as_index=False)
                .head(1)
                .sort_values(["num_ues", "num_sensors"])
            )
            best_by_ue_df = (
                df.sort_values("score")
                .groupby("num_ues", as_index=False)
                .head(1)
                .sort_values("num_ues")
            )
            best_by_ue_sensor_df.to_csv(
                base_dir / f"{method_name}_best_by_ue_sensor.csv",
                index=False,
            )
            best_by_ue_df.to_csv(base_dir / f"{method_name}_best_by_ue.csv", index=False)

            print("Score:", round(row["score"], 4))
            print("Saved partial results:", base_dir / f"{method_name}_all_results.csv")

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(base_dir / f"{method_name}_all_results.csv", index=False)
    best_by_ue_sensor_df = (
        final_df.sort_values("score")
        .groupby(["num_ues", "num_sensors"], as_index=False)
        .head(1)
        .sort_values(["num_ues", "num_sensors"])
    )
    best_by_ue_df = (
        final_df.sort_values("score")
        .groupby("num_ues", as_index=False)
        .head(1)
        .sort_values("num_ues")
    )
    best_by_ue_sensor_df.to_csv(base_dir / f"{method_name}_best_by_ue_sensor.csv", index=False)
    best_by_ue_df.to_csv(base_dir / f"{method_name}_best_by_ue.csv", index=False)

    print("\nFinished:", experiment_name)
    print("All results:", base_dir / f"{method_name}_all_results.csv")
    print("Best by UE/sensors:", base_dir / f"{method_name}_best_by_ue_sensor.csv")
    print("Best by UE overall:", base_dir / f"{method_name}_best_by_ue.csv")
    return final_df
