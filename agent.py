import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from utils import extract_steps, extract_tagged_answer
from prompts import SYSTEM_PROMPT, PRM_SYSTEM

# Qwen släppte aldrig en officiell PRM-3B. Officiella storlekar är 7B och 72B.
PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
SCORE_THRESHOLD = 0.2
MAX_RETRIES = 3
MAX_NEW_TOKENS = 512


def load_prm():
    """
    Ladda PRM:en separat (transformers + 4-bit). PRM är en token-classifier
    över <extra_0>-positioner, så vLLM ger ingen vinst här — vi kör den
    direkt via transformers med bitsandbytes-quant för att hålla VRAM nere.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        PRM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def score_steps(
    prm_model,
    prm_tokenizer,
    question: str,
    steps: list[str],
) -> list[float]:
    if not steps:
        return []

    messages = [
        {"role": "system", "content": PRM_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    input_text = prm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = prm_tokenizer(input_text, return_tensors="pt").to(prm_model.device)
    input_ids = inputs["input_ids"]

    step_sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>")
    token_masks = (input_ids == step_sep_id)  # [1, seq_len]

    with torch.no_grad():
        outputs = prm_model(**inputs)

    # logits shape: [1, seq_len, 2] — negative and positive class per position
    logits = outputs.logits
    step_logits = logits[token_masks]  # [num_steps, 2]
    scores = torch.softmax(step_logits.float(), dim=-1)[:, 1].tolist()

    if len(scores) != len(steps):
        raise ValueError(
            f"PRM returned {len(scores)} scores for {len(steps)} steps. "
            "Unexpected <extra_0> tokens found outside step content — "
            "check that question/system text does not contain this token."
        )

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


def _generate_vllm(model, tokenizer, prompt: str) -> tuple[str, int, int]:
    """
    Generera via Unsloths vLLM-backend (model.fast_generate). Faller tillbaka
    till model.generate om fast_generate inte finns (när fast_inference=False).
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    in_tok = input_ids.shape[1]

    if hasattr(model, "fast_generate"):
        outputs = model.fast_generate([prompt], sampling_params=sampling_params)
        completion = outputs[0].outputs[0].text
        out_tok = len(outputs[0].outputs[0].token_ids)
    else:
        # Fallback: model.generate (om vLLM inte är aktiverat)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        out_tok = gen.shape[1] - in_tok
        completion = tokenizer.decode(gen[0][in_tok:], skip_special_tokens=True)

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

        completion, in_tok, out_tok = _generate_vllm(model, tokenizer, prompt)
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
