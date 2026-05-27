import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import extract_steps, extract_tagged_answer
from prompts import SYSTEM_PROMPT, PRM_SYSTEM

# Qwen släppte aldrig en officiell PRM-3B. Officiella storlekar är 7B och 72B.
PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
SCORE_THRESHOLD = 0.2
MAX_RETRIES = 3
MAX_NEW_TOKENS = 512


def score_steps(prm_model, prm_tokenizer, question: str, steps: list[str]) -> list[float]:
    if not steps:
        return []

    import subprocess
    import json
    import sys

    # Förbered data som ska skickas till vår isolerade worker
    payload = {"question": question, "steps": steps}

    # Starta vår prm_worker i en helt ren miljö
    process = subprocess.Popen(
        ["python3", "/content/prm_worker.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Skicka data och vänta på svar
    stdout, stderr = process.communicate(input=json.dumps(payload))

    if process.returncode != 0:
        print(f"  [CRITICAL] PRM Worker failed: {stderr}", file=sys.stderr)
        raise ValueError(f"PRM Worker crashed with exit code {process.returncode}")

    # Läs av poängen som returnerades
    scores = json.loads(stdout.strip())
    return scores


def _format_initial_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _format_correction_prompt(question: str, accepted_steps: list[str], tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Prefill the start of the assistant turn with accepted steps
    prefill = "<think>" + "".join(f"<step>{s}</step>" for s in accepted_steps)
    return base + prefill


def _generate_hf(model, tokenizer, prompt):
    """
    Genererar text via Hugging Face-gränssnittet för Unsloth-modeller.
    Ersätter den gamla vLLM-logiken för att undvika sampling_params-krascher.
    """
    import torch
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,
            top_p=0.9
        )
        
    generated_tokens = outputs[0][len(inputs.input_ids[0]):]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    in_tok = len(inputs.input_ids[0])
    out_tok = len(outputs[0]) - in_tok
    
    return completion, in_tok, out_tok


def run_agentic_loop(
    model,
    tokenizer,
    prm_model,
    prm_tokenizer,
    question: str,
) -> dict:
    """
    Returns a dict with keys:
      status:       "confident" | "unsure" | "failed"
      answer:       str | None
      steps:        list[str]  (accepted steps; empty on "failed")
      total_tokens: int
    "unsure" additionally includes failing_idx: int (position where the PRM first rejected).
    """
    accepted_steps: list[str] = []
    total_tokens = 0
    best_fallback: dict | None = None

    for attempt in range(MAX_RETRIES):
        if accepted_steps:
            prompt = _format_correction_prompt(question, accepted_steps, tokenizer)
        else:
            prompt = _format_initial_prompt(question, tokenizer)

        completion, in_tok, out_tok = _generate_hf(model, tokenizer, prompt)
        total_tokens += in_tok + out_tok

        new_steps = extract_steps(completion)
        if not new_steps:
            continue

        all_steps = accepted_steps + new_steps

        try:
            scores = score_steps(prm_model, prm_tokenizer, question, all_steps)
        except ValueError as e:
            print(f"  [WARNING] score_steps failed (attempt {attempt}): {e} — skipping question")
            break

        # Find first failing step among the newly generated ones
        failing_idx = None
        for i, score in enumerate(scores[len(accepted_steps):], start=len(accepted_steps)):
            if score < SCORE_THRESHOLD:
                failing_idx = i
                break

        if failing_idx is None:
            # All new steps passed — only trust an answer from this validated chain
            answer = extract_tagged_answer(completion)
            if answer is not None:
                return {
                    "status": "confident",
                    "answer": answer,
                    "steps": all_steps,
                    "total_tokens": total_tokens,
                }
            # Steps passed but no answer tag yet — carry forward and retry
            accepted_steps = all_steps
        else:
            # PRM rejected this chain — track the best partial attempt as a fallback
            answer = extract_tagged_answer(completion)
            partial_steps = all_steps[:failing_idx]
            if answer is not None:
                if best_fallback is None or len(partial_steps) > len(best_fallback["steps"]):
                    best_fallback = {
                        "answer": answer,
                        "steps": partial_steps,
                        "failing_idx": failing_idx,
                    }
            accepted_steps = partial_steps

    if best_fallback is not None:
        return {
            "status": "unsure",
            "answer": best_fallback["answer"],
            "steps": best_fallback["steps"],
            "failing_idx": best_fallback["failing_idx"],
            "total_tokens": total_tokens,
        }

    return {
        "status": "failed",
        "answer": None,
        "steps": [],
        "total_tokens": total_tokens,
    }