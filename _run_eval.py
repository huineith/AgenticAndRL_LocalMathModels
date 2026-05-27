"""
Subprocess runner — kör EN evaluerings-körning och avsluta processen.

Designen är: en process per körning så att vLLM:s CUDA-kontext och NCCL-handles
rivs ner garanterat när processen exit:ar. Försök till in-process cleanup
mellan körningar är opålitligt på Colab.

Användning (kallas av benchmarker.py, inte direkt):
    python _run_eval.py --run-id 3b_10pct_no_agent --mode no_agent \\
        --checkpoint /path/to/ckpt --save-dir /path/to/results \\
        --n-questions 100 --seed 42

Mode:
    no_agent        — tränad checkpoint, generera direkt, extrahera <answer> tag
    agent           — tränad checkpoint + PRM, kör agentic loop
    math_baseline   — Qwen2.5-Math-7B-Instruct zero-shot, extrahera \\boxed{}

Exit codes:
    0   success — {run_id}_results.json skriven till save-dir
    !=0 fel av nåt slag (OOM, crash, missing checkpoint, ...)
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True,
                        help="Identifierar resultatfilen, t.ex. '3b_10pct_no_agent'")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["no_agent", "agent", "math_baseline", "untrained_baseline"])
    parser.add_argument("--checkpoint", type=str, default="",
                        help="For no_agent/agent: path to checkpoint. "
                             "For untrained_baseline: HuggingFace model ID. "
                             "Ignored for math_baseline (always uses Qwen2.5-Math-7B-Instruct).")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Imports inuti main() så att argparse-fel kan dyka upp utan GPU
    import json
    import torch
    from unsloth import FastLanguageModel

    from utils import (
        extract_tagged_answer,
        extract_boxed_answer,
        extract_h0_answer,
        extract_gsm8k_ground_truth,
        validate_format,
    )
    from agent import load_prm, run_agentic_loop
    from prompts import SYSTEM_PROMPT, MATH_BASELINE_SYSTEM
    from datasets import load_dataset

    MAX_SEQ_LENGTH = 1024
    MAX_GEN_TOKENS = 512
    BASELINE_MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"

    # ----------------------------------------------------- helpers (local)
    def load_questions(n: int, seed: int):
        ds = load_dataset("openai/gsm8k", "main")["test"]
        sampled = ds.shuffle(seed=seed).select(range(n))
        return sampled["question"], sampled["answer"]

    def format_trained_prompt(question, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )

    def format_math_baseline_prompt(question, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": MATH_BASELINE_SYSTEM},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )

    def generate_one(model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        in_tok = input_ids.shape[1]
        if hasattr(model, "fast_generate"):
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0.0, max_tokens=MAX_GEN_TOKENS, top_p=1.0)
            outputs = model.fast_generate([prompt], sampling_params=sp)
            completion = outputs[0].outputs[0].text
            out_tok = len(outputs[0].outputs[0].token_ids)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=MAX_GEN_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            out_tok = gen.shape[1] - in_tok
            completion = tokenizer.decode(gen[0][in_tok:], skip_special_tokens=True)
        return completion, in_tok, out_tok

    def load_trained(checkpoint, gpu_util):
        m, t = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            fast_inference=True,
            gpu_memory_utilization=gpu_util,
        )
        return m, t

    # ----------------------------------------------------- per-mode dispatch
    os.makedirs(args.save_dir, exist_ok=True)
    questions, solutions = load_questions(args.n_questions, args.seed)
    print(f"[{args.run_id}] Loaded {len(questions)} questions from GSM8K test split.", flush=True)

    if args.mode in ("no_agent", "agent", "untrained_baseline") and not args.checkpoint:
        print(f"[{args.run_id}] ERROR: --checkpoint required for mode={args.mode}",
              file=sys.stderr, flush=True)
        sys.exit(2)

    results = []

    # ............................................................ no_agent
    if args.mode == "no_agent":
        print(f"=== {args.run_id}: no-agent (trained checkpoint) ===", flush=True)
        model, tokenizer = load_trained(args.checkpoint, gpu_util=0.85)
        correct = 0
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            completion, in_tok, out_tok = generate_one(model, tokenizer,
                                                       format_trained_prompt(q, tokenizer))
            predicted = extract_tagged_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": validate_format(completion),
            })
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}",
                      flush=True)

    # ............................................................ agent
    elif args.mode == "agent":
        print(f"=== {args.run_id}: agent (trained checkpoint + PRM) ===", flush=True)
        model, tokenizer = load_trained(args.checkpoint, gpu_util=0.55)
        prm_model, prm_tokenizer = load_prm()

        partial_path = os.path.join(args.save_dir, f"{args.run_id}_partial.jsonl")
        completed: dict = {}
        if os.path.exists(partial_path):
            with open(partial_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        completed[r["question_idx"]] = r
            print(f"  Resuming: {len(completed)}/{len(questions)} already done.", flush=True)

        results_by_idx = [None] * len(questions)
        for idx, r in completed.items():
            results_by_idx[idx] = r
        correct = sum(1 for r in completed.values() if r["correct"])

        with open(partial_path, "a") as partial_f:
            for i, (q, gt) in enumerate(zip(questions, solutions)):
                if i in completed:
                    continue
                loop_result = run_agentic_loop(model, tokenizer, prm_model, prm_tokenizer, q)
                answer = loop_result["answer"]
                expected = extract_gsm8k_ground_truth(gt)
                ok = answer is not None and answer == expected
                if ok:
                    correct += 1
                # `format_ok` är inte meningsfullt för agent — den extraherar
                # ur sina egna steg snarare än att validera en hel completion.
                # Sätter True när vi har ett svar, annars False.
                r = {
                    "question_idx": i,
                    "question": q,
                    "predicted": answer,
                    "expected": expected,
                    "correct": ok,
                    "tokens": loop_result["total_tokens"],
                    "status": loop_result["status"],
                    "format_ok": answer is not None,
                }
                results_by_idx[i] = r
                partial_f.write(json.dumps(r) + "\n")
                partial_f.flush()
                if (i + 1) % 25 == 0:
                    done = sum(1 for x in results_by_idx[:i+1] if x is not None)
                    print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/done:.4f}",
                          flush=True)

        results = [r for r in results_by_idx if r is not None]
        if os.path.exists(partial_path):
            os.remove(partial_path)

    # ............................................................ untrained_baseline
    elif args.mode == "untrained_baseline":
        # HF model ID, ingen lokal checkpoint. Använder samma SYSTEM_PROMPT som
        # de tränade modellerna (oförändrad prompt), MEN extraktionen försöker
        # \boxed{} som fallback om <answer> saknas. Detta isolerar "kan modellen
        # räkna" från "kan modellen följa formatet". format_ok är fortfarande
        # strikt — kräver hela <think>...<answer> mönstret — så du kan se
        # exakt hur ofta de otränade modellerna ignorerar formatkraven.
        print(f"=== {args.run_id}: untrained_baseline ({args.checkpoint}) ===", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.checkpoint,   # HF ID
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            fast_inference=True,
            gpu_memory_utilization=0.85,
        )
        correct = 0
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            completion, in_tok, out_tok = generate_one(model, tokenizer,
                                                       format_trained_prompt(q, tokenizer))
            predicted = extract_h0_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": validate_format(completion),
            })
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}",
                      flush=True)

    # ............................................................ math_baseline
    elif args.mode == "math_baseline":
        print(f"=== {args.run_id}: math_baseline ({BASELINE_MODEL_ID}) ===", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASELINE_MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            fast_inference=True,
            gpu_memory_utilization=0.85,
        )
        correct = 0
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            completion, in_tok, out_tok = generate_one(model, tokenizer,
                                                       format_math_baseline_prompt(q, tokenizer))
            predicted = extract_boxed_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            # `format_ok` för math-baseline: hittade vi ett \boxed{}? Det är dess
            # tränings-format. Saknat \boxed{} = format violation.
            has_boxed = r"\boxed{" in completion
            results.append({
                "question": q,
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": has_boxed,
            })
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}",
                      flush=True)

    # ----------------------------------------------------- write JSON
    out_path = os.path.join(args.save_dir, f"{args.run_id}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"[{args.run_id}] DONE. Accuracy: {correct}/{total} = {correct/total:.4f}", flush=True)
    print(f"[{args.run_id}] Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
