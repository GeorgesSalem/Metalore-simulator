# MetaLore runner scripts

Copy these `.py` files into the root folder of `MetaLore-simulator`.

Recommended order:

```bash
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements_extra.txt
```

Create this folder:

```text
models/
```

Put your saved model here:

```text
models/ppo_model_repacked.zip
```

Run:

```bash
python check_env.py
python inspect_model.py
python evaluate_model.py
python continue_training.py
python plot_results.py
```

Open TensorBoard after training:

```bash
tensorboard --logdir results/tensorboard
```

The first scenario is set to `small` in `run_config.py` because the included `logs/jobs.csv` has 5 UEs and 8 sensors.
