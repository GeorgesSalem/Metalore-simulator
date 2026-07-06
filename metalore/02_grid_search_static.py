from __future__ import annotations

from static_reward_common import (
    OPERATOR_INTENTION,
    SENSOR_VALUES,
    UE_VALUES,
    build_experiment_name,
    run_training_evaluation_for_pairs,
)

# ============================================================
# STEP 2: grid search
# New static scenario:
# UEs = 20, 40, 60
# Sensors = 10, 20, 30
# Timesteps = 100k
# ============================================================

METHOD_NAME = "grid_ue20_40_60_s10_20_30"
EXPERIMENT_NAME = build_experiment_name(METHOD_NAME, sensor_values=SENSOR_VALUES)

# eta = synchronization discount factor
ETA_VALUES = [0.7, 0.8, 0.5]

# c2 = delay penalty
C2_VALUES = [-2.0, -3.0]

DRY_RUN_FIRST_N = None


def main():
    pairs_by_scenario = {}
    counter = 0
    stop = False

    for num_ues in UE_VALUES:
        for num_sensors in SENSOR_VALUES:
            pairs = []

            for eta in ETA_VALUES:
                for c2 in C2_VALUES:
                    counter += 1

                    if DRY_RUN_FIRST_N is not None and counter > DRY_RUN_FIRST_N:
                        stop = True
                        break

                    pairs.append((eta, c2))

                if stop:
                    break

            if pairs:
                pairs_by_scenario[(num_ues, num_sensors)] = pairs

            if stop:
                break

        if stop:
            break

    run_training_evaluation_for_pairs(
        method_name=METHOD_NAME,
        experiment_name=EXPERIMENT_NAME,
        pairs_by_scenario=pairs_by_scenario,
        operator_intention=OPERATOR_INTENTION,
    )


if __name__ == "__main__":
    main()
