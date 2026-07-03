from __future__ import annotations

from static_reward_common import (
    OPERATOR_INTENTION,
    UE_VALUES,
    build_experiment_name,
    run_training_evaluation_for_pairs,
)

# ============================================================
# STEP 2: grid search
# New static scenario:
# UEs = 10, 30, 50, 70, 90
# Sensors = 25
# Timesteps = 100k
# ============================================================

METHOD_NAME = "grid_ue10_30_50_70_90"
EXPERIMENT_NAME = build_experiment_name(METHOD_NAME)

# eta = synchronization discount factor
ETA_VALUES = [0.5, 0.7, 0.8, 0.9]

# c2 = delay penalty
C2_VALUES = [-5.0, -4.0, -3.0, -2.0]

DRY_RUN_FIRST_N = None


def main():
    pairs_by_ue = {}
    counter = 0

    for num_ues in UE_VALUES:
        pairs = []

        for eta in ETA_VALUES:
            for c2 in C2_VALUES:
                counter += 1

                if DRY_RUN_FIRST_N is not None and counter > DRY_RUN_FIRST_N:
                    break

                pairs.append((eta, c2))

            if DRY_RUN_FIRST_N is not None and counter >= DRY_RUN_FIRST_N:
                break

        if pairs:
            pairs_by_ue[num_ues] = pairs

        if DRY_RUN_FIRST_N is not None and counter >= DRY_RUN_FIRST_N:
            break

    run_training_evaluation_for_pairs(
        method_name=METHOD_NAME,
        experiment_name=EXPERIMENT_NAME,
        pairs_by_ue=pairs_by_ue,
        operator_intention=OPERATOR_INTENTION,
    )


if __name__ == "__main__":
    main()