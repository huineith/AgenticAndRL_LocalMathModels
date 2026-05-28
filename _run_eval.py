"""
_run_eval.py — Subprocess runner, kör EN evaluerings-körning och avslutar.

Fixade buggar:
  1. generate_one: skip_special_tokens=False + manuell EOS-strippning,
     do_sample=True  → samma fix som i agent.py.
  2. no_agent-prompt använder nu tokenizer.apply_chat_template (samma
     format som träningen) istället för den råa XML-strängen.
  3. args.n_questions (inte args.n_questions / args.n-questions namnkrock)
     — argparse-variabeln heter n_questions internt, används konsekvent.
  4. Resultat-listan appendas korrekt även vid agent-mode (var rätt, men
     explicit kontroll av res.get("steps", []) läggs till).
"""

import argparse
import os
import sys
import json
import torch
import random
from unsloth import FastLanguageModel
from datasets import load_dataset

from utils import (
    extract_tagged_answer,
    extract_boxed_answer,
    extract_gsm8k_ground_truth,
    extract_h0_answer,
)
from prompts import (
    SYSTEM_PROMPT,
    MATH_BASELINE_SYSTEM,
    MATH_BASELINE_TEMPLATE,
    UNTRAINED_CHAT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Generering (delas av untrained_baseline, no_agent, math_baseline)
# ---------------------------------------------------------------------------

def generate_one(model, tokenizer, prompt: str) -> tuple[str, int, int]:
    """
    FIX: skip_special_tokens=False — annars försvinner <answer>/<step>/<think>.
    FIX: do_sample=True — krävs för att temperature/top_p ska aktiveras.
    """
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = len(inputs.input_ids[0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_tokens = outputs[0][input_len:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    if tokenizer.eos_token:
        completion = completion.replace(tokenizer.eos_token, "")
    completion = completion.strip()

    in_tok = input_len
    out_tok = len(outputs[0]) - input_len
    return completion, in_tok, out_tok


# ---------------------------------------------------------------------------
# Huvud
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["no_agent", "agent", "math_baseline", "untrained_baseline"],
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[{args.run_id}] Starting runner mode={args.mode}", flush=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ Modell
    if args.mode in ("no_agent", "agent", "untrained_baseline"):
        if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
            print(f"[{args.run_id}] Detected LoRA adapter — resolving base model...", flush=True)
            if "3b" in args.run_id.lower() or "3b" in args.checkpoint.lower():
                base_model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
            else:
                base_model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

            print(f"[{args.run_id}] Loading base model {base_model_name}...", flush=True)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            model = FastLanguageModel.for_inference(model)
            print(f"[{args.run_id}] Applying LoRA adapter from {args.checkpoint}...", flush=True)
            model.load_adapter(args.checkpoint)
        else:
            print(f"[{args.run_id}] Loading pure base model: {args.checkpoint}", flush=True)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            model = FastLanguageModel.for_inference(model)

    elif args.mode == "math_baseline":
        print(f"[{args.run_id}] Loading SOTA Baseline: Qwen2.5-Math-7B-Instruct", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.for_inference(model)

    # ----------------------------------------------------------------- Dataset
    print(f"[{args.run_id}] Loading GSM8K test split...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")

    random.seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = indices[: args.n_questions]
    questions = [dataset[i]["question"] for i in selected]
    solutions = [dataset[i]["answer"] for i in selected]

    results = []
    correct = 0

    # ======================================================= 1. UNTRAINED BASELINE
    if args.mode == "untrained_baseline":
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            prompt = UNTRAINED_CHAT_TEMPLATE.format(question=q)
            completion, in_tok, out_tok = generate_one(model, tokenizer, prompt)
            predicted = extract_h0_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "completion": completion,
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": predicted is not None,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] acc: {correct/(i+1):.4f}", flush=True)

    # ============================================================== 2. NO_AGENT
    elif args.mode == "no_agent":
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            # FIX: använd apply_chat_template för konsekvent format med träningen
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            completion, in_tok, out_tok = generate_one(model, tokenizer, prompt)
            predicted = extract_tagged_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "completion": completion,
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": "<answer>" in completion,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] acc: {correct/(i+1):.4f}", flush=True)

    # ================================================================= 3. AGENT
    elif args.mode == "agent":
        from agent import run_agentic_loop
        print(f"[{args.run_id}] Agentic mode — PRM via subprocess worker", flush=True)

        for i, (q, sol) in enumerate(zip(questions, solutions)):
            res = run_agentic_loop(model, tokenizer, None, None, q)
            expected = extract_gsm8k_ground_truth(sol)
            ok = res["answer"] is not None and res["answer"] == expected
            if ok:
                correct += 1

            agent_chain = "\n".join(res.get("steps", []))
            results.append({
                "question": q,
                "completion": agent_chain,
                "predicted": res["answer"],
                "expected": expected,
                "correct": ok,
                "tokens": res["total_tokens"],
                "format_ok": res["status"] == "confident",
                "agent_status": res["status"],
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] acc: {correct/(i+1):.4f}", flush=True)

    # ======================================================= 4. MATH_BASELINE
    elif args.mode == "math_baseline":
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            prompt = MATH_BASELINE_TEMPLATE.format(
                system_prompt=MATH_BASELINE_SYSTEM,
                question=q,
            )
            completion, in_tok, out_tok = generate_one(model, tokenizer, prompt)
            predicted = extract_boxed_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "completion": completion,
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": r"\boxed{" in completion,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] acc: {correct/(i+1):.4f}", flush=True)

    # ------------------------------------------------------------------ Output
    out_path = os.path.join(args.save_dir, f"{args.run_id}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[{args.run_id}] DONE. Accuracy: {correct}/{len(results)} = {correct/len(results):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
