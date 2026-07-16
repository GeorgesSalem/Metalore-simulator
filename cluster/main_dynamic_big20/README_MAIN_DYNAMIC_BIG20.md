# One General PPO on Big20

## Final configuration

- One shared PPO model
- Fixed reward parameters: `eta = 0.8`, `C2 = -1`
- Normal dynamic training range:
  - UEs: 4 to 80, step 2
  - sensors: 4 to 40, step 2
- Stress-test pool: 100 UEs and 50 sensors
- Episode length: 1,000 simulator steps
- Load changes every 50 steps, giving 20 load phases per episode
- Maximum training budget: 1,000,000 total timesteps
- Checkpoint interval: approximately 100,000 timesteps
- Big20 resources: 24 simulator workers + 12 PyTorch threads
- Expected duration: roughly 30–90 minutes on Big20, but the smoke test is the reliable measurement

The 24 environments feed different trajectories to one shared PPO policy. They are not 24 PPO models.

## Capacity formulas

Communication is feasible when a bandwidth split exists such that:

`X*lambda_u*d_u/(B*q_u) + Y*lambda_s*d_s/(B*q_s) <= 1`

where `q = log2(1+SNR)` and `B = 600 MHz` in the simulator.

Computation is feasible when:

`X*lambda_u*c_u/F + Y*lambda_s*c_s/F <= 1`

where `F = 800` computation units.

With the current simulator parameters, the analytical estimates are approximately:

- communication: 54 devices at the worst map position
- communication: 64 devices at the conservative p10 channel
- communication: 85 devices at average channel quality
- computation: 114 devices
- empirical KPI/queue frontier from the dataset: around 80 total devices

Therefore 100 UEs + 50 sensors is a severe overload evaluation case, not a supported load.

## Install the package in the project root

From Windows PowerShell:

```powershell
scp C:\path\to\main_dynamic_ppo_big20_ready.zip salem@cluster.lip6.fr:~/MetaLore-simulator/
```

On the cluster login node:

```bash
cd ~/MetaLore-simulator
unzip -o main_dynamic_ppo_big20_ready.zip
chmod +x *_main_dynamic_big20.sh run_main_dynamic_big20.sh launch_main_dynamic_big20.sh check_main_dynamic_big20.sh stop_main_dynamic_big20.sh
```

## Enter the reserved node

```bash
OAR_JOB_ID=1304136 oarsh big20
cd ~/MetaLore-simulator
```

## Run the smoke test first

```bash
./quick_test_main_dynamic_big20.sh
```

The test writes to `main_dynamic_ppo_big20_test/` and does not interfere with the full output.

## Launch the full run and close the terminal safely

```bash
./launch_main_dynamic_big20.sh
```

The launcher uses `nohup`, writes a PID file and a timestamped log, and then returns. The SSH terminal/laptop may be closed.

## Check later

Reconnect to Big20 and run:

```bash
cd ~/MetaLore-simulator
./check_main_dynamic_big20.sh
```

Follow the live log:

```bash
LOG=$(cat main_dynamic_ppo_big20.logpath)
tail -f "$LOG"
```

`Ctrl+C` stops only the log display, not training.

## Resume after an interruption

Run the same launcher again:

```bash
./launch_main_dynamic_big20.sh
```

The full runner passes `--resume-auto`, so it resumes from the newest checkpoint in `main_dynamic_ppo_big20/`.

## Main outputs

- `main_dynamic_ppo_big20/models/main_dynamic_ppo_best.zip`
- `main_dynamic_ppo_big20/final_report.json`
- `main_dynamic_ppo_big20/capacity_report.json`
- `main_dynamic_ppo_big20/capacity_boundary.csv`
- `main_dynamic_ppo_big20/checkpoint_validation_summary.csv`
- `main_dynamic_ppo_big20/checkpoint_validation_episodes.csv`
- `main_dynamic_ppo_big20/dynamic_response_trace.csv`
- `main_dynamic_ppo_big20/plots/*.png`
- `results_backups/main_dynamic_ppo_big20_YYYYMMDD_HHMMSS.tar.gz`

The plots include capacity boundaries, training reward, checkpoint comparison, completion, AoRI/AoSI, queues, resource splits, dynamic load response and job flow.

## Download results to Windows

Run the included PowerShell script from the folder containing it:

```powershell
.\download_main_dynamic_big20.ps1
```

Or manually:

```powershell
mkdir C:\Users\LENOVO\Desktop\metalore_main_dynamic_results
scp -r salem@cluster.lip6.fr:~/MetaLore-simulator/main_dynamic_ppo_big20 C:\Users\LENOVO\Desktop\metalore_main_dynamic_results\
# The included PowerShell script also downloads the newest final tar.gz archive.
```
