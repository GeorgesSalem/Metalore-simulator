from __future__ import annotations

from static_reward_common import (
    OPERATOR_INTENTION,
    UE_VALUES,
    build_experiment_name,
    run_training_evaluation_for_pairs,
)

# ============================================================
# STEP 1: manual reward experiment
# Train one PPO per UE value using your chosen reward pair.
# Evaluation is on the SAME UE/sensor scenario used for training.
# ============================================================

METHOD_NAME = "manual"

# CHANGE ONLY THESE TWO VALUES for your chosen reward.
ETA = 0.8       # eta = synchronization discount factor
C2 = -2.0       # c2 = delay penalty

EXPERIMENT_NAME = build_experiment_name(METHOD_NAME, eta=ETA, c2=C2)


def main():
    pairs_by_ue = {
        num_ues: [(ETA, C2)]
        for num_ues in UE_VALUES
    }

    run_training_evaluation_for_pairs(
        method_name=METHOD_NAME,
        experiment_name=EXPERIMENT_NAME,
        pairs_by_ue=pairs_by_ue,
        operator_intention=OPERATOR_INTENTION,
    )


if __name__ == "__main__":
    main()
