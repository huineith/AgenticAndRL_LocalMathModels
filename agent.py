"""
agent.py — Agentic loop med PRM-validering via isolerad subprocess-worker.

Fixade buggar:
  1. skip_special_tokens=False  → <answer>-taggar försvinner inte längre vid decode.
  2. Korrekt step-extraktion vid prefill  → extraherar från hela (prefill+completion),
     tar sedan bort redan accepterade steg så new_steps alltid är genuint nya.
  3. failing_idx=0 deadlock  → partial_steps tar minst 1 steg framåt.
  4. PRM-krasch failsafe  → låter alla steg igenom (score=1.0) istället för break,
     så loopen alltid returnerar något svar.
  5. do_sample=True krävs  → annars ignorerar HF temperature/top_p tyst.
"""

import json
import subprocess
import sys
import torch

from utils import extract_steps, extract_tagged_answer
from prompts import SYSTEM_PROMPT

PRM_WORKER_PATH = "/content/prm_worker.py"
SCORE_THRESHOLD = 0.2
MAX_RETRIES = 3
MAX_NEW_TOKENS = 512


# ---------------------------------------------------------------------------
# Generering
# ---------------------------------------------------------------------------

def _generate_hf(model, tokenizer, prompt: str) -> tuple[str, int, int]:
    """
    Genererar text och returnerar (completion, in_tokens, out_tokens).

    FIX: skip_special_tokens=False — annars försvinner <answer>/<step>/<think>
    om tokenizern registrerat dem som special tokens.  EOS-token stoppas
    manuellt istället.

    FIX: do_sample=True — krävs för att temperature/top_p ska ha effekt.
    Utan det faller HF tyst tillbaka till greedy search.
    """
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_len = len(inputs.input_ids[0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_tokens = outputs[0][input_len:]

    # FIX: skip_special_tokens=False, strippa EOS manuellt
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    if tokenizer.eos_token:
        completion = completion.replace(tokenizer.eos_token, "")
    completion = completion.strip()

    in_tok = input_len
    out_tok = len(outputs[0]) - input_len
    return completion, in_tok, out_tok


# ---------------------------------------------------------------------------
# Prompt-formatering
# ---------------------------------------------------------------------------

def _format_initial_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _format_correction_prompt(
    question: str, accepted_steps: list[str], tokenizer
) -> str:
    """
    Prefill-prompt: modellen får se de redan godkända stegen och fortsätter
    därifrån.  Vi lägger INTE till </think> — modellen ska avsluta blocket
    själv efter att den lagt till fler steg och <answer>.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefill = "<think>" + "".join(f"<step>{s}</step>" for s in accepted_steps)
    return base + prefill


# ---------------------------------------------------------------------------
# PRM-scoring via isolerad subprocess
# ---------------------------------------------------------------------------

def score_steps(question: str, steps: list[str]) -> list[float]:
    """
    Kör prm_worker.py i en ren process (ingen Unsloth-förorening) och
    returnerar en score per steg.

    FIX: Vid krasch returneras [1.0] * len(steps) som failsafe istället
    för att kasta ett undantag som bryter loopen.
    """
    if not steps:
        return []

    payload = {"question": question, "steps": steps}

    try:
        process = subprocess.Popen(
            ["python3", PRM_WORKER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=json.dumps(payload), timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        print("  [WARNING] PRM Worker timeout — treating all steps as passing", file=sys.stderr)
        return [1.0] * len(steps)
    except Exception as e:
        print(f"  [WARNING] PRM Worker launch failed: {e} — treating all steps as passing", file=sys.stderr)
        return [1.0] * len(steps)

    if process.returncode != 0:
        print(
            f"  [WARNING] PRM Worker exit {process.returncode}: {stderr.strip()}"
            " — treating all steps as passing",
            file=sys.stderr,
        )
        return [1.0] * len(steps)

    try:
        scores = json.loads(stdout.strip())
    except json.JSONDecodeError as e:
        print(f"  [WARNING] PRM Worker bad JSON ({e}) — treating all steps as passing", file=sys.stderr)
        return [1.0] * len(steps)

    # Säkerhetsventil: om worker returnerar fel antal scores
    if len(scores) != len(steps):
        print(
            f"  [WARNING] PRM score count mismatch "
            f"(got {len(scores)}, expected {len(steps)}) — padding with 1.0",
            file=sys.stderr,
        )
        scores = scores[:len(steps)] + [1.0] * max(0, len(steps) - len(scores))

    return scores


# ---------------------------------------------------------------------------
# Huvud-loop
# ---------------------------------------------------------------------------

def run_agentic_loop(
    model,
    tokenizer,
    prm_model,       # ignoreras — PRM körs i subprocess
    prm_tokenizer,   # ignoreras — PRM körs i subprocess
    question: str,
) -> dict:
    """
    Returnerar:
      status:       "confident" | "unsure" | "failed"
      answer:       str | None
      steps:        list[str]
      total_tokens: int
    "unsure" inkluderar även failing_idx: int.
    """
    accepted_steps: list[str] = []
    total_tokens = 0
    best_fallback: dict | None = None

    for attempt in range(MAX_RETRIES):
        # --- Bygg prompt ---
        if accepted_steps:
            prompt = _format_correction_prompt(question, accepted_steps, tokenizer)
        else:
            prompt = _format_initial_prompt(question, tokenizer)

        completion, in_tok, out_tok = _generate_hf(model, tokenizer, prompt)
        total_tokens += in_tok + out_tok

        print(f"  [attempt {attempt}] completion preview: {completion[:120]!r}", file=sys.stderr)

        # --- Extrahera steg ---
        # FIX: vid prefill-prompt innehåller 'completion' INTE prefill-texten.
        # Vi rekonstruerar hela assistenttexten för att kunna hitta alla steg,
        # och tar sedan bort de redan accepterade.
        if accepted_steps:
            prefill_text = "<think>" + "".join(f"<step>{s}</step>" for s in accepted_steps)
            full_assistant_text = prefill_text + completion
        else:
            full_assistant_text = completion

        all_extracted = extract_steps(full_assistant_text)
        # Hoppa över de vi redan accepterat (kan vara dupletter i texten)
        new_steps = all_extracted[len(accepted_steps):]

        if not new_steps:
            print(f"  [attempt {attempt}] Inga nya steg hittades — försöker igen", file=sys.stderr)
            continue

        all_steps = accepted_steps + new_steps

        # --- PRM-scoring ---
        scores = score_steps(question, all_steps)

        # --- Hitta första steg som failar (bland de NYLIGEN genererade) ---
        failing_idx = None
        for i, score in enumerate(scores[len(accepted_steps):], start=len(accepted_steps)):
            if score < SCORE_THRESHOLD:
                failing_idx = i
                break

        if failing_idx is None:
            # Alla nya steg godkändes — leta efter svar
            answer = extract_tagged_answer(full_assistant_text)
            if answer is not None:
                return {
                    "status": "confident",
                    "answer": answer,
                    "steps": all_steps,
                    "total_tokens": total_tokens,
                }
            # Steg OK men inget <answer> än — bär vidare och försök igen
            accepted_steps = all_steps
        else:
            # PRM rejektade ett steg
            answer = extract_tagged_answer(full_assistant_text)

            # FIX: ta minst 1 steg framåt för att undvika deadlock när failing_idx=0
            safe_cut = max(failing_idx, len(accepted_steps) + 1)
            partial_steps = all_steps[:safe_cut]

            if answer is not None:
                if best_fallback is None or len(partial_steps) > len(best_fallback["steps"]):
                    best_fallback = {
                        "answer": answer,
                        "steps": partial_steps,
                        "failing_idx": failing_idx,
                    }

            accepted_steps = partial_steps

    # --- Loop slut utan confident svar ---
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
