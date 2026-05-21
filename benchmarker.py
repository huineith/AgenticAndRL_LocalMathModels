import gc
import os
import csv
import json
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from unsloth import FastLanguageModel

from utils import extract_tagged_answer, extract_last_integer, extract_gsm8k_ground_truth
from agent import load_prm, run_agentic_loop
from prompts import SYSTEM_PROMPT, BASELINE_PROMPT_TEMPLATE

BASELINE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LENGTH = 1024
MAX_GEN_TOKENS = 512

GROUP_META = [
    {"id": 1, "name": "3B No Agent",   "size": "3b", "agentic": False, "color": "steelblue",  "marker": "o"},
    {"id": 2, "name": "3B + Agent",    "size": "3b", "agentic": True,  "color": "steelblue",  "marker": "D"},
    {"id": 3, "name": "7B No Agent",   "size": "7b", "agentic": False, "color": "seagreen",   "marker": "o"},
    {"id": 4, "name": "7B + Agent",    "size": "7b", "agentic": True,  "color": "seagreen",   "marker": "D"},
    {"id": 5, "name": "14B Baseline",  "size": "14b","agentic": False, "color": "darkorange", "marker": "s"},
]


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def load_benchmark_questions(n: int = 500, seed: int = 42):
    dataset = load_dataset("openai/gsm8k", "main")
    test_split = dataset["test"]
    sampled = test_split.shuffle(seed=seed).select(range(n))
    questions = sampled["question"]
    solutions = sampled["answer"]
    return questions, solutions


def _format_trained_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _format_baseline_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": BASELINE_PROMPT_TEMPLATE.format(question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    accuracy = correct / total
    avg_tokens = total_tokens / total
    tokens_per_correct = total_tokens / correct if correct > 0 else float("inf")
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "tokens_per_correct": tokens_per_correct,
        "correct": correct,
        "total": total,
    }


def _group_result_path(save_dir: str, gid: int) -> str:
    return os.path.join(save_dir, f"group_{gid}_results.json")


def _save_group(save_dir: str, gid: int, results: list[dict]):
    path = _group_result_path(save_dir, gid)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"  Checkpoint saved: {path}")


def _load_group(save_dir: str, gid: int) -> list[dict] | None:
    path = _group_result_path(save_dir, gid)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _run_agentic_with_recovery(
    model,
    tokenizer,
    prm_model,
    prm_tokenizer,
    questions: list[str],
    solutions: list[str],
    partial_path: str,
) -> list[dict]:
    # Reload any questions already processed in a prior interrupted run
    completed: dict[int, dict] = {}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    completed[r["question_idx"]] = r
        print(f"  Resuming: {len(completed)}/{len(questions)} questions already done.")

    results: list[dict | None] = [None] * len(questions)
    for idx, r in completed.items():
        results[idx] = r
    correct = sum(1 for r in completed.values() if r["correct"])

    with open(partial_path, "a") as partial_f:
        for i, (question, gt_solution) in enumerate(zip(questions, solutions)):
            if i in completed:
                continue

            loop_result = run_agentic_loop(model, tokenizer, prm_model, prm_tokenizer, question)
            answer = loop_result["answer"]
            expected = extract_gsm8k_ground_truth(gt_solution)
            is_correct = answer is not None and answer == expected
            if is_correct:
                correct += 1

            result = {
                "question_idx": i,
                "question": question,
                "predicted": answer,
                "expected": expected,
                "correct": is_correct,
                "tokens": loop_result["total_tokens"],
                "status": loop_result["status"],
            }
            results[i] = result
            partial_f.write(json.dumps(result) + "\n")
            partial_f.flush()

            if (i + 1) % 50 == 0:
                done = sum(1 for r in results[: i + 1] if r is not None)
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/done:.4f}")

    return [r for r in results if r is not None]


def _run_nonagentic(model, tokenizer, questions, solutions, prompt_fn, extract_fn) -> list[dict]:
    results = []
    correct = 0
    for i, (question, solution) in enumerate(zip(questions, solutions)):
        prompt = prompt_fn(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        in_tok = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        out_tok = outputs.shape[1] - in_tok
        completion = tokenizer.decode(outputs[0][in_tok:], skip_special_tokens=True)
        predicted = extract_fn(completion)
        expected = extract_gsm8k_ground_truth(solution)
        is_correct = predicted is not None and predicted == expected
        if is_correct:
            correct += 1
        results.append({
            "question": question,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
            "tokens": in_tok + out_tok,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}")
    return results


def _load_trained_model(checkpoint_path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_benchmark(
    checkpoint_3b: str,
    checkpoint_7b: str,
    save_dir: str,
    n_questions: int = 500,
    seed: int = 42,
):
    os.makedirs(save_dir, exist_ok=True)

    # Fail fast if save_dir lives on Drive but Drive isn't mounted
    if save_dir.startswith("/content/drive") and not os.path.exists("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive does not appear to be mounted. "
            "Run drive.mount('/content/drive') in your notebook (Cell 2) before calling run_benchmark()."
        )

    questions, solutions = load_benchmark_questions(n_questions, seed)
    print(f"Loaded {len(questions)} benchmark questions from GSM8K test split.")

    all_results = {}  # group_id -> list[dict]
    summaries = {}    # group_id -> metrics dict

    # ------------------------------------------------------------------ Group 1
    print("\n=== Group 1: 3B No Agent ===")
    cached = _load_group(save_dir, 1)
    if cached is not None:
        all_results[1] = cached
        summaries[1] = _compute_metrics(cached)
        print(f"  Loaded cached results ({len(cached)} questions). Accuracy: {summaries[1]['accuracy']:.4f}")
    else:
        model, tokenizer = _load_trained_model(checkpoint_3b)
        results = _run_nonagentic(model, tokenizer, questions, solutions,
                                  _format_trained_prompt, extract_tagged_answer)
        del model, tokenizer
        clear_gpu()
        all_results[1] = results
        summaries[1] = _compute_metrics(results)
        _save_group(save_dir, 1, results)
        print(f"  Accuracy: {summaries[1]['accuracy']:.4f}  Avg tokens: {summaries[1]['avg_tokens']:.1f}")

    # ------------------------------------------------------------------ Group 2
    print("\n=== Group 2: 3B + Agent ===")
    cached = _load_group(save_dir, 2)
    if cached is not None:
        all_results[2] = cached
        summaries[2] = _compute_metrics(cached)
        print(f"  Loaded cached results ({len(cached)} questions). Accuracy: {summaries[2]['accuracy']:.4f}")
    else:
        model, tokenizer = _load_trained_model(checkpoint_3b)
        prm_model, prm_tokenizer = load_prm()
        partial_path = os.path.join(save_dir, "group_2_partial.jsonl")
        results = _run_agentic_with_recovery(model, tokenizer, prm_model, prm_tokenizer,
                                             questions, solutions, partial_path)
        del model, tokenizer, prm_model, prm_tokenizer
        clear_gpu()
        all_results[2] = results
        summaries[2] = _compute_metrics(results)
        _save_group(save_dir, 2, results)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"  Accuracy: {summaries[2]['accuracy']:.4f}  Avg tokens: {summaries[2]['avg_tokens']:.1f}")

    # ------------------------------------------------------------------ Group 3
    print("\n=== Group 3: 7B No Agent ===")
    cached = _load_group(save_dir, 3)
    if cached is not None:
        all_results[3] = cached
        summaries[3] = _compute_metrics(cached)
        print(f"  Loaded cached results ({len(cached)} questions). Accuracy: {summaries[3]['accuracy']:.4f}")
    else:
        model, tokenizer = _load_trained_model(checkpoint_7b)
        results = _run_nonagentic(model, tokenizer, questions, solutions,
                                  _format_trained_prompt, extract_tagged_answer)
        del model, tokenizer
        clear_gpu()
        all_results[3] = results
        summaries[3] = _compute_metrics(results)
        _save_group(save_dir, 3, results)
        print(f"  Accuracy: {summaries[3]['accuracy']:.4f}  Avg tokens: {summaries[3]['avg_tokens']:.1f}")

    # ------------------------------------------------------------------ Group 4
    print("\n=== Group 4: 7B + Agent ===")
    cached = _load_group(save_dir, 4)
    if cached is not None:
        all_results[4] = cached
        summaries[4] = _compute_metrics(cached)
        print(f"  Loaded cached results ({len(cached)} questions). Accuracy: {summaries[4]['accuracy']:.4f}")
    else:
        model, tokenizer = _load_trained_model(checkpoint_7b)
        prm_model, prm_tokenizer = load_prm()
        partial_path = os.path.join(save_dir, "group_4_partial.jsonl")
        results = _run_agentic_with_recovery(model, tokenizer, prm_model, prm_tokenizer,
                                             questions, solutions, partial_path)
        del model, tokenizer, prm_model, prm_tokenizer
        clear_gpu()
        all_results[4] = results
        summaries[4] = _compute_metrics(results)
        _save_group(save_dir, 4, results)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"  Accuracy: {summaries[4]['accuracy']:.4f}  Avg tokens: {summaries[4]['avg_tokens']:.1f}")

    # ------------------------------------------------------------------ Group 5
    print("\n=== Group 5: 14B Baseline ===")
    cached = _load_group(save_dir, 5)
    if cached is not None:
        all_results[5] = cached
        summaries[5] = _compute_metrics(cached)
        print(f"  Loaded cached results ({len(cached)} questions). Accuracy: {summaries[5]['accuracy']:.4f}")
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASELINE_MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        results = _run_nonagentic(model, tokenizer, questions, solutions,
                                  _format_baseline_prompt, extract_last_integer)
        del model, tokenizer
        clear_gpu()
        all_results[5] = results
        summaries[5] = _compute_metrics(results)
        _save_group(save_dir, 5, results)
        print(f"  Accuracy: {summaries[5]['accuracy']:.4f}  Avg tokens: {summaries[5]['avg_tokens']:.1f}")

    # ----------------------------------------------------------------- Save CSV
    summary_path = os.path.join(save_dir, "benchmark_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group_id", "group_name", "accuracy", "avg_tokens",
            "tokens_per_correct", "correct", "total"
        ])
        writer.writeheader()
        for meta in GROUP_META:
            gid = meta["id"]
            writer.writerow({"group_id": gid, "group_name": meta["name"], **summaries[gid]})

    detailed_path = os.path.join(save_dir, "benchmark_detailed.csv")
    with open(detailed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group_id", "group_name", "question_idx", "question",
            "predicted", "expected", "correct", "tokens", "status"
        ])
        writer.writeheader()
        for meta in GROUP_META:
            gid = meta["id"]
            for i, r in enumerate(all_results[gid]):
                writer.writerow({
                    "group_id": gid,
                    "group_name": meta["name"],
                    "question_idx": i,
                    "question": r.get("question", ""),
                    "predicted": r["predicted"],
                    "expected": r["expected"],
                    "correct": r["correct"],
                    "tokens": r["tokens"],
                    "status": r.get("status", ""),
                })

    plot_path = os.path.join(save_dir, "benchmark_plot.png")
    _plot_results(summaries, plot_path)

    print(f"\nResults saved to {save_dir}")
    print(f"  Summary : {summary_path}")
    print(f"  Detailed: {detailed_path}")
    print(f"  Plot    : {plot_path}")
    return summaries


def _plot_results(summaries: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))

    for meta in GROUP_META:
        gid = meta["id"]
        s = summaries[gid]
        x = s["avg_tokens"]
        y = s["accuracy"] * 100
        ax.scatter(x, y, color=meta["color"], marker=meta["marker"], s=120, zorder=3)
        ax.annotate(
            f"{meta['name']}\n{y:.1f}%  |  {s['tokens_per_correct']:.0f} tok/ans",
            xy=(x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
        )

    # Legend for model size
    size_patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",  markersize=9, label="3B"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",   markersize=9, label="7B"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", markersize=9, label="14B (baseline)"),
    ]
    mode_patches = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=9, label="No agent"),
        plt.Line2D([0], [0], marker="D", color="grey", linestyle="None", markersize=9, label="+ Agent"),
        plt.Line2D([0], [0], marker="s", color="grey", linestyle="None", markersize=9, label="Zero-shot"),
    ]
    ax.legend(handles=size_patches + mode_patches, fontsize=8, loc="lower right")

    ax.set_xlabel("Average tokens per question (input + output)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Compute Cost — GSM8K (500 questions)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")
