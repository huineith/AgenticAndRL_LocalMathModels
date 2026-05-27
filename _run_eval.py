"""
Subprocess runner — kör EN evaluerings-körning och avsluta processen.
Safe layout för A100 med dynamisk LoRA-detektering och fullständig råtext-lagring.
"""
import argparse
import os
import sys
import json
import torch
import random
from unsloth import FastLanguageModel
from datasets import load_dataset

# Importera alla dina extraktorer från utils.py
from utils import extract_tagged_answer, extract_boxed_answer, extract_gsm8k_ground_truth, extract_h0_answer

# Importera alla prompter och mallar från prompts.py
from prompts import (
    SYSTEM_PROMPT, 
    MATH_BASELINE_SYSTEM, 
    BASELINE_PROMPT_TEMPLATE, 
    MATH_BASELINE_TEMPLATE,
    UNTRAINED_CHAT_TEMPLATE
)

def generate_one(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,
            top_p=0.9
        )
    completion = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    in_tok = len(inputs.input_ids[0])
    out_tok = len(outputs[0]) - in_tok
    return completion, in_tok, out_tok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["no_agent", "agent", "math_baseline", "untrained_baseline"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[{args.run_id}] Starting runner mode={args.mode}", flush=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- DYNAMISK OCH SÄKER MODELLADDNING FÖR UNSLOTH / LoRA ---
    if args.mode in ("no_agent", "agent", "untrained_baseline"):
        
        # Om checkpointen är en lokal lora-adapter (innehåller adapter_config.json)
        if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
            print(f"[{args.run_id}] Detected LoRA adapter directory. Resolving base model size...", flush=True)
            if "3b" in args.run_id or "3b" in args.checkpoint.lower():
                base_model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
            else:
                base_model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
            
            print(f"[{args.run_id}] Loading base model {base_model_name} with 4bit...", flush=True)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=2048,
                load_in_4bit=True
            )
            print(f"[{args.run_id}] Applying tränade LoRA-adapters från {args.checkpoint}...", flush=True)
            model = FastLanguageModel.for_inference(model)
            model.load_adapter(args.checkpoint)
        else:
            # Annars ladda som en ren basmodell (t.ex. unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit)
            print(f"[{args.run_id}] Loading pure base model: {args.checkpoint}", flush=True)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint,
                max_seq_length=2048,
                load_in_4bit=True
            )
            model = FastLanguageModel.for_inference(model)
    
    elif args.mode == "math_baseline":
        print(f"[{args.run_id}] Loading SOTA Baseline: Qwen/Qwen2.5-Math-7B-Instruct", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True
        )
        model = FastLanguageModel.for_inference(model)

    # --- LADDA DATASET (HuggingFace GSM8K) ---
    print(f"[{args.run_id}] Loading GSM8K test split from HuggingFace...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")
    
    random.seed(args.seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:args.n_questions]
    questions = [dataset[i]["question"] for i in selected_indices]
    solutions = [dataset[i]["answer"] for i in selected_indices]

    results = []
    correct = 0

    # --- EVALUERINGSLOOP ---
# ==================================================== 1. UNTRAINED BASELINE
    if args.mode == "untrained_baseline":
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            # MODULÄR PROMPT: Använder nu samma chatt-struktur (<messages>) som H1 och H2
            prompt = UNTRAINED_CHAT_TEMPLATE.format(question=q)
            
            completion, in_tok, out_tok = generate_one(model, tokenizer, prompt)
            predicted = extract_h0_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "completion": completion,  # Sparar basmodellens råtext
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": predicted is not None,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}", flush=True)

    # ============================================================ 2. NO_AGENT
    elif args.mode == "no_agent":
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            prompt = f"<messages><message role='system'>{SYSTEM_PROMPT}</message><message role='user'>{q}</message></messages>"
            completion, in_tok, out_tok = generate_one(model, tokenizer, prompt)
            predicted = extract_tagged_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            results.append({
                "question": q,
                "completion": completion,  # Sparar GRPO-modellens råtext
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": "<answer>" in completion,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}", flush=True)

    # =============================================================== 3. AGENT
# =============================================================== 3. AGENT
    elif args.mode == "agent":
        # Vi importerar enbart run_agentic_loop nu eftersom PRM styrs inuti agent.py
        from agent import run_agentic_loop
        print(f"[{args.run_id}] Initializing Agentic Mode with isolated PRM worker...", flush=True)

        for i, (q, sol) in enumerate(zip(questions, solutions)):
            # Vi skickar None, None på PRM-platserna eftersom din nya agent.py 
            # sköter all PRM-hantering självständigt via sin worker!
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
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}", flush=True)
    # ======================================================= 4. MATH_BASELINE
    elif args.mode == "math_baseline":
        for i, (q, sol) in enumerate(zip(questions, solutions)):
            # MODULÄR PROMPT: Läser nu MATH_BASELINE_TEMPLATE och MATH_BASELINE_SYSTEM från prompts.py
            prompt = MATH_BASELINE_TEMPLATE.format(
                system_prompt=MATH_BASELINE_SYSTEM,
                question=q
            )
            completion, in_tok, out_tok = generate_one(model, tokenizer, prompt)
            predicted = extract_boxed_answer(completion)
            expected = extract_gsm8k_ground_truth(sol)
            ok = predicted is not None and predicted == expected
            if ok:
                correct += 1
            has_boxed = r"\boxed{" in completion
            results.append({
                "question": q,
                "completion": completion,  # Sparar SOTA-baslinjens råtext
                "predicted": predicted,
                "expected": expected,
                "correct": ok,
                "tokens": in_tok + out_tok,
                "format_ok": has_boxed,
            })
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] Running accuracy: {correct/(i+1):.4f}", flush=True)

    # ----------------------------------------------------- Skriv utdata-JSON
    out_path = os.path.join(args.save_dir, f"{args.run_id}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    
    print(f"[{args.run_id}] DONE. Accuracy: {correct}/{len(results)} = {correct/len(results):.4f}", flush=True)

if __name__ == "__main__":
    main()