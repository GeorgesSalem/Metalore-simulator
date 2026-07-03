from __future__ import annotations

from static_reward_common import (
    OPERATOR_INTENTION,
    build_experiment_name,
    run_training_evaluation_for_pairs,
)



METHOD_NAME = "llm"
EXPERIMENT_NAME = build_experiment_name(METHOD_NAME)

# Replace this example with the exact LLM_CANDIDATES returned by ChatGPT.
LLM_CANDIDATES = {
    10: [(0.6, -3.0), (0.8, -4.0), (0.4, -2.0)],
    30: [(0.8, -3.0), (0.9, -2.0), (0.85, -4.0)],
    50: [(0.6, -3.5), (0.8, -1.5), (0.7, -2.5)],
    70: [(0.7, -3.0), (0.5, -4.0), (0.8, -2.0)],
    90: [(0.9, -2.0), (0.8, -2.0), (0.7, -3.0)],
}



def main():
    run_training_evaluation_for_pairs(
        method_name=METHOD_NAME,
        experiment_name=EXPERIMENT_NAME,
        pairs_by_ue=LLM_CANDIDATES,
        operator_intention=OPERATOR_INTENTION,
    )


if __name__ == "__main__":
    main()
