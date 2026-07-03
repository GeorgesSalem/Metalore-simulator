from __future__ import annotations

import ast
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

from static_reward_common import (
    METALORE_DIR,
    OPERATOR_INTENTION,
    UE_VALUES,
    NUM_SENSORS,
    TRAIN_TIMESTEPS,
    build_experiment_name,
)


# ============================================================
# LOCAL LLM SETTINGS
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"

GRID_METHOD_NAME = "grid_ue10_30_50_70_90"

# Your real grid result file name
GRID_FILENAME = "grid_ue10_30_50_70_90_all_results.csv"

TOP_N_PER_UE = 8

ETA_MIN = 0.1
ETA_MAX = 1.0
C2_MIN = -5.0
C2_MAX = 0.0

MIN_CANDIDATES = 2
MAX_CANDIDATES = 3


# ============================================================
# PATHS
# ============================================================

BASE_DIR = METALORE_DIR / "results" / "reward_pipeline" / "static_25s"

GRID_EXPERIMENT = build_experiment_name(GRID_METHOD_NAME)
GRID_DIR = BASE_DIR / GRID_EXPERIMENT

OUTPUT_DIR = BASE_DIR / f"local_llm_candidates_{GRID_METHOD_NAME}_{TRAIN_TIMESTEPS // 1000}k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_DIR = OUTPUT_DIR / "prompts_by_ue"
RAW_DIR = OUTPUT_DIR / "raw_responses_by_ue"

PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_JSON = OUTPUT_DIR / "llm_candidates.json"
CANDIDATES_PY = OUTPUT_DIR / "llm_candidates_generated.py"


# ============================================================
# GRID CSV LOADING
# ============================================================

def find_grid_csv() -> Path:
    candidates = [
        GRID_DIR / GRID_FILENAME,
        GRID_DIR / "grid_all_results.csv",
    ]

    if GRID_DIR.exists():
        candidates.extend(sorted(GRID_DIR.glob("*all_results.csv")))

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find grid result CSV.\n"
        f"Checked folder:\n{GRID_DIR}\n\n"
        f"Expected one of:\n"
        f"- {GRID_FILENAME}\n"
        f"- grid_all_results.csv"
    )


GRID_ALL_CSV = find_grid_csv()


def load_grid_results() -> pd.DataFrame:
    df = pd.read_csv(GRID_ALL_CSV)

    required = [
        "num_ues",
        "num_sensors",
        "eta",
        "c2",
        "mean_aori",
        "mean_aosi",
        "job_completion_rate",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in grid CSV: {missing}")

    return df


def get_top_rows_for_ue(df: pd.DataFrame, ue: int) -> pd.DataFrame:
    sub = df[df["num_ues"] == ue].copy()

    if sub.empty:
        raise ValueError(f"No rows found for UE={ue}")

    if "score" in sub.columns:
        sub = sub.sort_values("score", ascending=True)
    else:
        sub = sub.sort_values(
            ["job_completion_rate", "mean_aori", "mean_aosi"],
            ascending=[False, True, True],
        )

    return sub.head(TOP_N_PER_UE)


# ============================================================
# PROMPT
# ============================================================

def make_ue_prompt(ue: int, top_df: pd.DataFrame) -> str:
    cols = [
        "num_ues",
        "num_sensors",
        "eta",
        "c2",
        "score",
        "mean_aori",
        "mean_aosi",
        "job_completion_rate",
        "throughput_jobs_per_step",
        "jobs_generated",
        "jobs_transmitted",
        "jobs_processed",
    ]

    cols = [c for c in cols if c in top_df.columns]
    grid_text = top_df[cols].round(4).to_csv(index=False)

    return f"""

You are choosing reward hyperparameters for PPO.

Return ONLY valid JSON.
No explanation.
No reasoning.
No markdown.
No text before or after JSON.

Required JSON format:
{{
  "pairs": [[0.5, -2.0], [0.7, -3.0]]
}}

Task:
Choose 2 or 3 candidate pairs for this scenario:
- UEs: {ue}
- Sensors: {NUM_SENSORS}
- Training timesteps: {TRAIN_TIMESTEPS}
- Operator intention: user priority = minimize mean_aori while keeping job_completion_rate high.

Allowed ranges:
- eta between {ETA_MIN} and {ETA_MAX}
- c2 between {C2_MIN} and {C2_MAX}

Selection rule:
1. Prefer job_completion_rate >= 0.99.
2. If no row reaches 0.99, do NOT assume the data is wrong.
3. If no row reaches 0.99, choose the highest completion candidates.
4. Then choose low mean_aori.
5. Then keep mean_aosi reasonable.
6. Prefer candidates near the best grid-search region.
7. You may suggest intermediate values inside the allowed ranges.

Top grid-search rows:
{grid_text}

Return JSON now.
""".strip()


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt: str) -> str:
    """
    First tries Ollama JSON mode.
    If Qwen returns empty, retries without JSON mode.
    """

    url = "http://localhost:11434/api/chat"
    last_error = None

    for use_json_format in [True, False]:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. "
                        "No explanation. No markdown. No reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 300,
            },
        }

        if use_json_format:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))

            content = result.get("message", {}).get("content", "")
            content = content.strip()

            if content:
                return content

        except Exception as e:
            last_error = e

    if last_error is not None:
        print("Warning: Ollama call failed:", last_error)

    return ""


# ============================================================
# PARSING AND VALIDATION
# ============================================================

def remove_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_pairs_from_object(obj) -> list[tuple[float, float]]:
    pairs = []

    if isinstance(obj, dict):
        # Best expected format: {"pairs": [[eta, c2], ...]}
        for key in ["pairs", "candidates", "candidate_pairs", "values"]:
            if key in obj:
                pairs.extend(extract_pairs_from_object(obj[key]))

        # Format: {"eta": 0.5, "c2": -2.0}
        if "eta" in obj and "c2" in obj:
            pairs.append((float(obj["eta"]), float(obj["c2"])))

        # Search nested objects
        for value in obj.values():
            pairs.extend(extract_pairs_from_object(value))

    elif isinstance(obj, list):
        # Format: [0.5, -2.0]
        if len(obj) == 2:
            try:
                eta = float(obj[0])
                c2 = float(obj[1])
                pairs.append((eta, c2))
                return pairs
            except Exception:
                pass

        for item in obj:
            pairs.extend(extract_pairs_from_object(item))

    return pairs


def parse_pairs_from_text(text: str) -> list[tuple[float, float]]:
    text = remove_think_blocks(text)

    if not text:
        return []

    text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()

    # Try JSON
    try:
        obj = json.loads(text)
        return extract_pairs_from_object(obj)
    except Exception:
        pass

    # Try extracting JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)

        try:
            obj = json.loads(candidate)
            return extract_pairs_from_object(obj)
        except Exception:
            pass

        try:
            obj = ast.literal_eval(candidate)
            return extract_pairs_from_object(obj)
        except Exception:
            pass

    # Try regex pairs like (0.5, -2.0) or [0.5, -2.0]
    regex = r"[\(\[]\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\)\]]"
    matches = re.findall(regex, text)

    pairs = []
    for eta_str, c2_str in matches:
        pairs.append((float(eta_str), float(c2_str)))

    return pairs


def validate_pairs(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = []

    for eta, c2 in pairs:
        try:
            eta = round(float(eta), 3)
            c2 = round(float(c2), 3)
        except Exception:
            continue

        if not (ETA_MIN <= eta <= ETA_MAX):
            continue

        if not (C2_MIN <= c2 <= C2_MAX):
            continue

        pair = (eta, c2)

        if pair not in clean:
            clean.append(pair)

    return clean[:MAX_CANDIDATES]


def fallback_from_grid(top_df: pd.DataFrame, existing: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pairs = list(existing)

    for _, row in top_df.iterrows():
        eta = round(float(row["eta"]), 3)
        c2 = round(float(row["c2"]), 3)
        pair = (eta, c2)

        if pair not in pairs:
            pairs.append(pair)

        if len(pairs) >= MIN_CANDIDATES:
            break

    return pairs[:MAX_CANDIDATES]


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(candidates: dict[int, list[tuple[float, float]]]):
    json_ready = {
        str(ue): [[eta, c2] for eta, c2 in pairs]
        for ue, pairs in candidates.items()
    }

    with open(CANDIDATES_JSON, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=4)

    lines = ["LLM_CANDIDATES = {"]

    for ue in UE_VALUES:
        pairs_text = ", ".join(f"({eta}, {c2})" for eta, c2 in candidates[ue])
        lines.append(f"    {ue}: [{pairs_text}],")

    lines.append("}")

    py_text = "\n".join(lines)

    with open(CANDIDATES_PY, "w", encoding="utf-8") as f:
        f.write(py_text + "\n")

    print("\n=== LOCAL LLM CANDIDATES ===")
    print(py_text)

    print("\nSaved:")
    print(CANDIDATES_JSON)
    print(CANDIDATES_PY)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== LOCAL LLM CANDIDATE GENERATION WITH OLLAMA ===")
    print("Model:", OLLAMA_MODEL)
    print("Grid experiment:", GRID_EXPERIMENT)
    print("Grid CSV:", GRID_ALL_CSV)

    df = load_grid_results()

    final_candidates = {}

    for ue in UE_VALUES:
        print("\n================================================")
        print(f"Processing UE = {ue}")
        print("================================================")

        top_df = get_top_rows_for_ue(df, ue)
        prompt = make_ue_prompt(ue, top_df)

        prompt_path = PROMPTS_DIR / f"prompt_ue_{ue}.txt"
        raw_path = RAW_DIR / f"raw_ue_{ue}.txt"

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        raw = call_ollama(prompt)

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw)

        parsed_pairs = parse_pairs_from_text(raw)
        clean_pairs = validate_pairs(parsed_pairs)

        if len(clean_pairs) < MIN_CANDIDATES:
            print(f"Warning: UE {ue}: local LLM output was empty or invalid. Using grid fallback.")
            clean_pairs = fallback_from_grid(top_df, clean_pairs)
        else:
            print(f"UE {ue}: parsed local LLM candidates:", clean_pairs)

        final_candidates[ue] = clean_pairs

    save_outputs(final_candidates)


if __name__ == "__main__":
    main()