from pathlib import Path

# Root should be the MetaLore-simulator folder when you copy these scripts there.
ROOT = Path(__file__).resolve().parent

# Put your SB3 model here:
MODEL_PATH = ROOT / "models" / "ppo_40ue_20s_good_100kv1.zip"
CONTINUED_MODEL_PATH = ROOT / "models" / "ppo_50ue_30s_bad_reward_50k.zip"
NEW_MODEL_PATH = ROOT / "models" / "ppo_40ue_20s_good_100kv1.zip"

RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"
 
# Recommended first scenario because your logs/jobs.csv shows 5 UEs and 8 sensors.
# Options: "default", "small", "large", "mobile_sensor", "multi_cell","high_density","high_density_multicell"
SCENARIO = "high_density"

# Training/evaluation parameters
EVAL_EPISODES = 50
TRAIN_TIMESTEPS = 100_000

# PPO parameters when training a new model from zero
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
N_EPOCHS = 10
SEED = 5555
