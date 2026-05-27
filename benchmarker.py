"""
Benchmark-orchestrator. Två separata funktioner — en per hypotes:

  run_h1_benchmark(checkpoints, save_dir, ...)
    Data efficiency × model size. Kör 6 no-agent-körningar:
      checkpoints = {
          "1_5b_2pct":  "/path/.../grpo_1_5b_2pct_best",
          "1_5b_5pct":  "/path/...",
          "1_5b_10pct": "/path/...",
          "3b_2pct":    "/path/...",
          "3b_5pct":    "/path/...",
          "3b_10pct":   "/path/...",
      }

  run_h2_benchmark(agent_checkpoint, save_dir, ...)
    Agentic compute vs SOTA. Kör 2 körningar:
      - agent på vald checkpoint (default 3B 10%)
      - Math-7B-Instruct zero-shot baseline

Varje körning sker i en separat subprocess för vLLM-cleanup-safety.
Subprocess-krasch → CalledProcessError raisas omedelbart, ingen CSV skrivs.

Per save_dir produceras:
    {run_id}_results.json       per körning
    benchmark_summary.csv       en rad per run_id
    benchmark_detailed.csv      en rad per fråga

Run-ID:n är fria strängar utan grupp-IDs. CSV:erna sorteras i ordningen
som körningarna kördes / hittades.
"""
import os
import sys
import csv
import json
import subprocess
from typing import Optional

# Officiell math-tunad baseline. Qwen2.5-Math finns bara i 1.5B/7B/72B.
BASELINE_MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
H2_AGENT_RUN_ID = "3b_10pct_agent"
H2_BASELINE_RUN_ID = "math_7b_baseline"

# H0: untrained instruct-modeller (samma basmodeller som tränades). Använder
# samma SYSTEM_PROMPT som tränade modeller → direkt jämförbar format-compliance.
H0_UNTRAINED_RUNS = {
    "1_5b_untrained": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b_untrained":   "Qwen/Qwen2.5-3B-Instruct",
}

# Hitta _run_eval.py bredvid denna fil
_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_run_eval.py")


# ============================================================================
# Subprocess wrapper
# ============================================================================

def _run_subprocess(run_id: str, mode: str, save_dir: str,
                    n_questions: int, seed: int,
                    checkpoint: Optional[str] = None) -> None:
    """Kör runner-skriptet som subprocess. Vidarebefordrar stdout/stderr live.
    Raisear CalledProcessError om subprocess inte exit:ar med 0."""
    cmd = [
        sys.executable, _RUNNER,
        "--run-id", run_id,
        "--mode", mode,
        "--save-dir", save_dir,
        "--n-questions", str(n_questions),
        "--seed", str(seed),
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])

    print(f"\n>>> Launching subprocess for run_id={run_id}, mode={mode}", flush=True)
    print(f">>> Command: {' '.join(cmd)}", flush=True)

    # check=True raisear CalledProcessError vid nonzero exit code.
    # stdout/stderr ärvs från parent → live-output i notebooken.
    subprocess.run(cmd, check=True)


def _result_path(save_dir: str, run_id: str) -> str:
    return os.path.join(save_dir, f"{run_id}_results.json")


def _execute_run(run_id: str, mode: str, save_dir: str,
                 n_questions: int, seed: int,
                 checkpoint: Optional[str] = None) -> None:
    """Cache + subprocess-utförande. Hoppar över om resultatfilen redan finns."""
    out_path = _result_path(save_dir, run_id)
    if os.path.exists(out_path):
        print(f"\n=== {run_id} — CACHED ===")
        print(f"  Found existing {out_path}, skipping subprocess.")
        return

    if mode in ("no_agent", "agent"):
        if not checkpoint:
            raise ValueError(f"Run {run_id} requires a checkpoint path (mode={mode}).")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint for {run_id} not found: {checkpoint}")
    elif mode == "untrained_baseline":
        if not checkpoint:
            raise ValueError(f"Run {run_id} requires a HuggingFace model ID (mode={mode}).")
        # checkpoint är ett HF ID som "Qwen/Qwen2.5-1.5B-Instruct" — kan inte
        # filsystem-valideras. Subprocess kommer fail:a tydligt om ID är fel.

    try:
        _run_subprocess(run_id, mode, save_dir, n_questions, seed, checkpoint)
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Run {run_id} subprocess failed with exit code {e.returncode}.", flush=True)
        if e.returncode in (-9, 137):
            print("!!! Likely OOM-kill. Lower gpu_memory_utilization in _run_eval.py.", flush=True)
        print("!!! Stopping benchmark — no summary CSV will be written.", flush=True)
        raise

    if not os.path.exists(out_path):
        raise RuntimeError(
            f"Run {run_id} subprocess exited cleanly but {out_path} was not written. "
            "Something is wrong inside _run_eval.py."
        )


# ============================================================================
# Metrics
# ============================================================================

def _compute_metrics(results: list[dict], is_agent: bool) -> dict:
    """Räkna ut alla per-grupp-metrics från en lista per-fråge-resultat."""
    total = len(results)
    if total == 0:
        return {}
    correct = sum(1 for r in results if r["correct"])
    total_tokens = sum(r["tokens"] for r in results)
    extracted = sum(1 for r in results if r["predicted"] is not None)
    format_ok = sum(1 for r in results if r.get("format_ok", False))

    metrics = {
        "accuracy": correct / total,
        "avg_tokens": total_tokens / total,
        "tokens_per_correct": (total_tokens / correct) if correct > 0 else float("inf"),
        "correct": correct,
        "total": total,
        # När modellen *gav* ett svar, hur ofta var det rätt?
        "answer_extracted_rate": extracted / total,
        "answer_extracted_accuracy": (correct / extracted) if extracted > 0 else 0.0,
        # Format compliance — sekundärt intresse. För no_agent: matchade
        # <think>...<answer>-mönstret. För math_baseline: hittades \boxed{}.
        # För untrained_baseline: matchade strikta <think>...<answer>-mönstret.
        "format_violation_rate": 1.0 - (format_ok / total),
    }

    if is_agent:
        # Status-fördelning
        from collections import Counter
        status_counts = Counter(r.get("status", "missing") for r in results)
        for status in ("confident", "unsure", "failed"):
            metrics[f"status_{status}_rate"] = status_counts.get(status, 0) / total

        # PRM-kalibrering: accuracy betingat på status
        for status in ("confident", "unsure"):
            in_status = [r for r in results if r.get("status") == status]
            if in_status:
                metrics[f"accuracy_when_{status}"] = (
                    sum(1 for r in in_status if r["correct"]) / len(in_status)
                )
            else:
                metrics[f"accuracy_when_{status}"] = 0.0
    return metrics


# CSV-kolumnordning. Mode-specifika fält kommer sist, tomma där de inte gäller.
_SUMMARY_FIELDS = [
    "run_id", "mode", "checkpoint",
    "accuracy", "avg_tokens", "tokens_per_correct",
    "correct", "total",
    "answer_extracted_rate", "answer_extracted_accuracy",
    "format_violation_rate",
    # H2-specifikt
    "status_confident_rate", "status_unsure_rate", "status_failed_rate",
    "accuracy_when_confident", "accuracy_when_unsure",
]

_DETAILED_FIELDS = [
    "run_id", "mode", "question_idx", "question",
    "predicted", "expected", "correct", "tokens", "format_ok", "status",
]


def _write_summary_csv(save_dir: str, ordered_runs: list[dict]) -> str:
    """ordered_runs: list of {run_id, mode, checkpoint, metrics}."""
    path = os.path.join(save_dir, "benchmark_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for run in ordered_runs:
            row = {
                "run_id": run["run_id"],
                "mode": run["mode"],
                "checkpoint": run["checkpoint"],
            }
            row.update({k: run["metrics"].get(k, "") for k in _SUMMARY_FIELDS
                        if k not in row})
            writer.writerow(row)
    return path


def _write_detailed_csv(save_dir: str, ordered_runs: list[dict],
                        results_by_run: dict[str, list[dict]]) -> str:
    path = os.path.join(save_dir, "benchmark_detailed.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DETAILED_FIELDS)
        writer.writeheader()
        for run in ordered_runs:
            for i, r in enumerate(results_by_run[run["run_id"]]):
                writer.writerow({
                    "run_id": run["run_id"],
                    "mode": run["mode"],
                    "question_idx": r.get("question_idx", i),
                    "question": r.get("question", ""),
                    "predicted": r["predicted"],
                    "expected": r["expected"],
                    "correct": r["correct"],
                    "tokens": r["tokens"],
                    "format_ok": r.get("format_ok", ""),
                    "status": r.get("status", ""),
                })
    return path


def _aggregate_and_write(save_dir: str, run_plan: list[dict]) -> dict:
    """Läs alla {run_id}_results.json, räkna metrics, skriv CSV-er.
    run_plan: list of {run_id, mode, checkpoint} i önskad CSV-ordning.
    Returnerar {run_id: metrics_dict}."""
    summaries = {}
    results_by_run = {}
    for run in run_plan:
        with open(_result_path(save_dir, run["run_id"])) as f:
            results = json.load(f)
        results_by_run[run["run_id"]] = results
        summaries[run["run_id"]] = _compute_metrics(results, is_agent=(run["mode"] == "agent"))

    ordered_runs = [{**run, "metrics": summaries[run["run_id"]]} for run in run_plan]
    summary_path = _write_summary_csv(save_dir, ordered_runs)
    detailed_path = _write_detailed_csv(save_dir, ordered_runs, results_by_run)

    print(f"\nResults written to {save_dir}")
    print(f"  Summary : {summary_path}")
    print(f"  Detailed: {detailed_path}")
    return summaries


# ============================================================================
# H0: Untrained baselines (no GRPO, same base models)
# ============================================================================

def run_h0_benchmark(
    save_dir: str,
    n_questions: int = 100,
    seed: int = 42,
    runs: Optional[dict] = None,
) -> dict:
    """
    Kör H0: otränade Qwen2.5-Instruct (1.5B + 3B) med din SYSTEM_PROMPT.
    Detta är 0%-data-baseline för H1 — visar vad modellen kan utan GRPO.

    Med SYSTEM_PROMPT (kräver <think><step>...</step></think><answer>) förväntas
    accuracy vara LÅG och format_violation_rate HÖG. Det är poängen: format-
    compliance och accuracy växer båda fram med GRPO-träning, och H0 är 0%-
    punkten på den kurvan.

    runs: dict {run_id: hf_model_id}. Default = H0_UNTRAINED_RUNS.

    Returnerar {run_id: metrics_dict}.
    """
    os.makedirs(save_dir, exist_ok=True)
    _check_drive_mounted(save_dir)
    if not os.path.exists(_RUNNER):
        raise FileNotFoundError(f"Runner script not found at {_RUNNER}.")

    if runs is None:
        runs = H0_UNTRAINED_RUNS

    run_plan = [
        {"run_id": run_id, "mode": "untrained_baseline", "checkpoint": hf_id}
        for run_id, hf_id in runs.items()
    ]

    for run in run_plan:
        _execute_run(run["run_id"], run["mode"], save_dir, n_questions, seed,
                     checkpoint=run["checkpoint"])

    return _aggregate_and_write(save_dir, run_plan)


# ============================================================================
# H1: Data efficiency × model size
# ============================================================================

# Förväntade nycklar i checkpoints-dict. Behåller deterministisk ordning i CSV.
H1_EXPECTED_RUN_IDS = [
    "1_5b_2pct", "1_5b_5pct", "1_5b_10pct",
    "3b_2pct",   "3b_5pct",   "3b_10pct",
]


def run_h1_benchmark(
    checkpoints: dict,
    save_dir: str,
    n_questions: int = 100,
    seed: int = 42,
) -> dict:
    """
    Kör H1: 6 no-agent-körningar (1.5B och 3B, vardera vid 2.5/5/10% av träningsdata).

    checkpoints: dict från run-id till checkpoint-sökväg. Förväntade nycklar:
        1_5b_2pct, 1_5b_5pct, 1_5b_10pct, 3b_2pct, 3b_5pct, 3b_10pct
        (notera: 2pct för 2.5%-checkpointen — matchar int(0.025*100)=2 från
        grpo_trainer.py). Saknade nycklar accepteras (för partiella körningar).

    Returnerar {run_id: metrics_dict}.
    """
    os.makedirs(save_dir, exist_ok=True)
    _check_drive_mounted(save_dir)

    if not os.path.exists(_RUNNER):
        raise FileNotFoundError(
            f"Runner script not found at {_RUNNER}. "
            "Make sure _run_eval.py lives next to benchmarker.py."
        )

    # Bygg run plan i deterministisk ordning
    run_plan = []
    for run_id in H1_EXPECTED_RUN_IDS:
        if run_id not in checkpoints:
            print(f"Note: '{run_id}' not in checkpoints dict, skipping.")
            continue
        run_plan.append({
            "run_id": run_id,
            "mode": "no_agent",
            "checkpoint": checkpoints[run_id],
        })

    if not run_plan:
        raise ValueError("No checkpoints provided to run_h1_benchmark.")

    # Kör alla
    for run in run_plan:
        _execute_run(run["run_id"], run["mode"], save_dir, n_questions, seed,
                     checkpoint=run["checkpoint"])

    return _aggregate_and_write(save_dir, run_plan)


# ============================================================================
# H2: Agentic compute vs SOTA baseline
# ============================================================================

def run_h2_benchmark(
    agent_checkpoint: str,
    save_dir: str,
    n_questions: int = 100,
    seed: int = 42,
    skip_baseline: bool = False,
) -> dict:
    """
    Kör H2: agent på vald checkpoint + Math-7B-Instruct zero-shot baseline.

    agent_checkpoint: sökväg till checkpointen agenten ska köras på. Default
        antagandet är 3B (10%) — sätt manuellt när du har H1-resultaten.

    skip_baseline=True hoppar över baseline-körningen (användbart om du redan
    har kört den i en annan run och vill spara tid).

    Returnerar {run_id: metrics_dict}.
    """
    os.makedirs(save_dir, exist_ok=True)
    _check_drive_mounted(save_dir)

    if not os.path.exists(_RUNNER):
        raise FileNotFoundError(f"Runner script not found at {_RUNNER}.")

    if not agent_checkpoint:
        raise ValueError("agent_checkpoint cannot be empty.")
    if not os.path.exists(agent_checkpoint):
        raise FileNotFoundError(f"agent_checkpoint not found: {agent_checkpoint}")

    run_plan = [
        {
            "run_id": H2_AGENT_RUN_ID,
            "mode": "agent",
            "checkpoint": agent_checkpoint,
        },
    ]
    if not skip_baseline:
        run_plan.append({
            "run_id": H2_BASELINE_RUN_ID,
            "mode": "math_baseline",
            "checkpoint": "",
        })

    for run in run_plan:
        _execute_run(run["run_id"], run["mode"], save_dir, n_questions, seed,
                     checkpoint=run["checkpoint"] or None)

    return _aggregate_and_write(save_dir, run_plan)


# ============================================================================
# Helpers
# ============================================================================

def _check_drive_mounted(save_dir: str):
    if save_dir.startswith("/content/drive") and not os.path.exists("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive does not appear to be mounted. "
            "Run drive.mount('/content/drive') in your notebook before calling the benchmark."
        )
