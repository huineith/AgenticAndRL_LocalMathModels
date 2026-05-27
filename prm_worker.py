# prm_worker.py
import sys
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from prompts import PRM_SYSTEM

def main():
    # Läs in argumenten som skickades från agenten via stdin
    input_data = json.loads(sys.stdin.read())
    question = input_data["question"]
    steps = input_data["steps"]
    
    PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"
    SCORE_THRESHOLD = 0.2

    # Ladda tokenizer och modell helt rent utan Unsloth-påverkan
    tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_ID, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False
    )
    model.eval()

    messages = [
        {"role": "system", "content": PRM_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    step_sep_id = tokenizer.convert_tokens_to_ids("<extra_0>")
    token_masks = (input_ids == step_sep_id)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    step_logits = logits[token_masks]
    scores = torch.softmax(step_logits.float(), dim=-1)[:, 1].tolist()

    # Skicka tillbaka poängen till huvudprocessen via stdout
    print(json.dumps(scores))

if __name__ == "__main__":
    main()