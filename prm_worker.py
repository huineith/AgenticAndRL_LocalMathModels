# prm_worker.py
import sys
import json
import os
import torch

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prompts import PRM_SYSTEM
except ImportError:
    PRM_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    # --- Läs input ---
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw)
        question = input_data["question"]
        steps = input_data["steps"]
    except Exception as e:
        print(f"[prm_worker] Input parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if not steps:
        print(json.dumps([]))
        sys.exit(0)

    PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"

    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_ID, trust_remote_code=True)

        # FIX: AutoModel + trust_remote_code=True laddar Qwen2ForProcessRewardModel
        # vilket ger logits shape [1, seq_len, 2] istället för [1, 2]
        model = AutoModel.from_pretrained(
            PRM_MODEL_ID,
            dtype=torch.bfloat16,   # FIX: dtype istället för torch_dtype (undviker varning)
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"[prm_worker] Model load error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        messages = [
            {"role": "system", "content": PRM_SYSTEM},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "<extra_0>".join(steps) + "<extra_0>",
            },
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(input_text, return_tensors="pt")

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        step_sep_id = tokenizer.convert_tokens_to_ids("<extra_0>")
        if step_sep_id == tokenizer.unk_token_id:
            print("[prm_worker] WARNING: <extra_0> resolved to UNK", file=sys.stderr)

        token_masks = (input_ids == step_sep_id)
        n_separators = token_masks.sum().item()

        if n_separators == 0:
            print("[prm_worker] ERROR: No <extra_0> separators found", file=sys.stderr)
            sys.exit(1)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                past_key_values=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        # FIX: logits har nu rätt shape [1, seq_len, 2] — indexera med mask på dim 1
        logits = outputs.logits          # [1, seq_len, 2]
        step_logits = logits[0][token_masks[0]]  # [n_steps, 2]
        scores = torch.softmax(step_logits.float(), dim=-1)[:, 1].tolist()

        if len(scores) != len(steps):
            print(
                f"[prm_worker] WARNING: got {len(scores)} scores for {len(steps)} steps",
                file=sys.stderr,
            )
            scores = scores[:len(steps)] + [1.0] * max(0, len(steps) - len(scores))

    except Exception as e:
        print(f"[prm_worker] Scoring error: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(scores))
    sys.exit(0)


if __name__ == "__main__":
    main()