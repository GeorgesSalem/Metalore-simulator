STATIC REWARD PIPELINE — CLEAN VERSION
======================================

COPY THESE FILES INTO:
C:\Users\LENOVO\Desktop\MetaLore-simulator\metalore

Files:
1. static_reward_common.py
2. 01_manual_reward_static.py
3. 02_grid_search_static.py
4. 03_make_llm_prompt_from_grid.py
5. 04_llm_candidates_static.py
6. 05_plot_static_reward_results.py

GOAL
----
Sensors fixed = 25.
UE values = 10, 20, 30, 40, 50.
Each PPO model is trained alone on one UE/sensor scenario and evaluated on the same scenario.
PPO hyperparameters are fixed.
Only reward hyperparameters eta and c2 are changed.
eta = config["reward"]["discount_factor"]
c2  = config["reward"]["delay_penalty"]

STEP 1 — Manual reward pair
---------------------------
Open 01_manual_reward_static.py and set:
ETA = 0.7
C2 = -2.0

Run:
python 01_manual_reward_static.py

Results:
results\reward_pipeline\static_25s\manual_static_25s_eta0p7_c2m2p0_user_priority_50k

STEP 2 — Grid search
--------------------
Open 02_grid_search_static.py and set ETA_VALUES and C2_VALUES if needed.
Run:
python 02_grid_search_static.py

Results:
results\reward_pipeline\static_25s\grid_static_25s_user_priority_50k

Important: This is many trainings. Start with TRAIN_TIMESTEPS = 50_000 in static_reward_common.py.
For a quick test, set DRY_RUN_FIRST_N = 2 in 02_grid_search_static.py.

STEP 3 — Make LLM prompt
------------------------
Run:
python 03_make_llm_prompt_from_grid.py

It creates a text file:
results\reward_pipeline\static_25s\grid_static_25s_user_priority_50k\llm_prompt_user_priority.txt

Open it, copy everything, paste it into ChatGPT.
ChatGPT will return LLM_CANDIDATES.

STEP 4 — LLM candidates
-----------------------
Open 04_llm_candidates_static.py.
Replace the LLM_CANDIDATES dictionary with the one ChatGPT returned.
Run:
python 04_llm_candidates_static.py

Results:
results\reward_pipeline\static_25s\llm_static_25s_user_priority_50k

STEP 5 — Plot
-------------
Run:
python 05_plot_static_reward_results.py

Plots:
results\reward_pipeline\static_25s\plots_user_priority_manual_vs_grid_vs_llm

Plots include:
- AoRI grouped by PPO1..PPO5
- AoSI grouped by PPO1..PPO5
- Throughput grouped by PPO1..PPO5
- Completion rate
- Training time
- Estimated convergence timestep
- Jobs generated/transmitted/processed

WHERE TO CHANGE SETTINGS
------------------------
static_reward_common.py:
- UE_VALUES
- NUM_SENSORS
- TRAIN_TIMESTEPS
- EVAL_EPISODES
- OPERATOR_INTENTION

01_manual_reward_static.py:
- ETA
- C2

02_grid_search_static.py:
- ETA_VALUES
- C2_VALUES

04_llm_candidates_static.py:
- LLM_CANDIDATES returned by ChatGPT
