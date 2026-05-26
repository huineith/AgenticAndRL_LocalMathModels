import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from rewards import compute_reward
from utils import extract_tagged_answer, extract_gsm8k_ground_truth
from prompts import SYSTEM_PROMPT


MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
VAL_INDICES = list(range(7373, 7473))


def load_gsm8k_splits(data_fraction: float, seed: int = 42):
    dataset = load_dataset("openai/gsm8k", "main")
    train_full = dataset["train"]
    n = int(len(train_full) * data_fraction)
    assert n < VAL_INDICES[0], (
        f"Training slice ({n} examples) overlaps with validation indices "
        f"(starting at {VAL_INDICES[0]}). Reduce data_fraction below "
        f"{VAL_INDICES[0] / len(train_full):.2f}."
    )
    train_slice = train_full.select(range(n))
    val_slice = train_full.select(VAL_INDICES)
    return train_slice, val_slice


def format_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def prepare_grpo_dataset(train_slice) -> Dataset:
    def map_fn(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": example["question"]},
            ],
            "solution": example["answer"],
        }
    return train_slice.map(map_fn, remove_columns=train_slice.column_names)


def evaluate_on_validation(model, tokenizer, val_slice) -> float:
    was_training = model.training
    FastLanguageModel.for_inference(model)
    model.eval()
    correct = 0
    try:
        with torch.no_grad():
            for example in val_slice:
                prompt = format_prompt(example["question"], tokenizer)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                completion = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                predicted = extract_tagged_answer(completion)
                expected = extract_gsm8k_ground_truth(example["answer"])
                if predicted is not None and predicted == expected:
                    correct += 1
        return correct / len(val_slice)
    finally:
        # Återställ träningsläge så GRPO-loopen kan fortsätta.
        FastLanguageModel.for_training(model)
        if was_training:
            model.train()


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, model, tokenizer, val_slice, save_path: str, patience: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.val_slice = val_slice
        self.save_path = save_path
        self.patience = patience
        self.best_accuracy = 0.0
        self.no_improve = 0
        # Historik för plottning: (epoch, val_accuracy) per epoch.
        self.accuracy_history: list[dict] = []
        self.best_epoch = 0

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch = int(state.epoch)
        accuracy = evaluate_on_validation(self.model, self.tokenizer, self.val_slice)
        print(f"[Epoch {epoch}] Val accuracy: {accuracy:.4f}  Best: {self.best_accuracy:.4f}")

        self.accuracy_history.append({"epoch": epoch, "val_accuracy": accuracy})

        epoch_path = f"{self.save_path}_epoch{epoch}"
        self.model.save_pretrained(epoch_path)
        self.tokenizer.save_pretrained(epoch_path)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            self.no_improve = 0
            self.model.save_pretrained(self.save_path + "_best")
            self.tokenizer.save_pretrained(self.save_path + "_best")
            print(f"  -> New best saved to {self.save_path}_best")
        else:
            self.no_improve += 1
            print(f"  -> No improvement ({self.no_improve}/{self.patience})")
            if self.no_improve >= self.patience:
                print("  -> Early stopping triggered.")
                control.should_training_stop = True

        return control


def load_base_model(model_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,,
        load_in_4bit=True,
        max_lora_rank=LORA_RANK,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def run_sft_warmup(model, tokenizer, warmup_path: str, save_path: str):
    with open(warmup_path) as f:
        examples = json.load(f)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    required_keys = {"question", "response"}
    for i, ex in enumerate(examples):
        missing = required_keys - ex.keys()
        if missing:
            raise ValueError(f"warmup_data.json entry {i} missing keys: {missing}")

    texts = []
    for ex in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

    raw_dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        # result["labels"] = result["input_ids"].copy()
        return result

    tokenized = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=save_path + "_warmup_tmp",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            report_to="none",
        ),
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(save_path + "_warmup")
    tokenizer.save_pretrained(save_path + "_warmup")

    # Säkerställ att modellen är i träningsläge inför GRPO-fasen.
    FastLanguageModel.for_training(model)
    model.train()
    print(f"SFT warmup complete -> {save_path}_warmup")


def save_training_metrics(trainer, callback, save_path: str, meta: dict):
    """
    Spara allt vi behöver för att plotta GRPO-träningen i efterhand.

    Skriver två filer bredvid checkpointen:
      {save_path}_metrics.json  — log_history (reward, reward_std, kl, loss per
                                  loggsteg) + val_accuracy per epoch + metadata.
      {save_path}_metrics.png   — snabb översiktsplot (reward + loss + val-acc).

    log_history kommer från TRL/HF-tränaren och innehåller posterna som annars
    bara skrevs till stdout. Vi dumpar dem råa så att du kan plotta vad du vill
    senare utan att köra om träningen.
    """
    log_history = trainer.state.log_history

    metrics = {
        "meta": meta,
        "best_val_accuracy": callback.best_accuracy,
        "best_epoch": callback.best_epoch,
        "val_accuracy_history": callback.accuracy_history,
        "log_history": log_history,
    }

    json_path = f"{save_path}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {json_path}")

    # Översiktsplot. Misslyckas tyst om något saknas — JSON är det viktiga.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [e["step"] for e in log_history if "reward" in e]
        rewards = [e["reward"] for e in log_history if "reward" in e]
        loss_steps = [e["step"] for e in log_history if "loss" in e]
        losses = [e["loss"] for e in log_history if "loss" in e]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(steps, rewards, color="seagreen")
        axes[0].set_title("Mean reward")
        axes[0].set_xlabel("step")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(loss_steps, losses, color="steelblue")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("step")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        ep = [h["epoch"] for h in callback.accuracy_history]
        acc = [h["val_accuracy"] * 100 for h in callback.accuracy_history]
        axes[2].plot(ep, acc, marker="o", color="darkorange")
        axes[2].set_title("Validation accuracy")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("%")
        axes[2].grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"GRPO {meta.get('size_tag', '')} {meta.get('pct', '')}%")
        plt.tight_layout()
        png_path = f"{save_path}_metrics.png"
        plt.savefig(png_path, dpi=130)
        plt.close()
        print(f"  Plot saved: {png_path}")
    except Exception as e:
        print(f"  [WARNING] Could not render metrics plot: {e} (JSON still saved)")


def run_grpo(
    model_name: str,
    data_fraction: float,
    save_dir: str,
    warmup_path: str = "warmup_data.json",
    max_epochs: int = 3,
    seed: int = 42,
):
    pct = int(data_fraction * 100)
    if "3B" in model_name:
        size_tag = "3b"
    elif "7B" in model_name:
        size_tag = "7b"
    else:
        raise ValueError(
            f"Cannot infer model size from name '{model_name}'. "
            "Expected '3B' or '7B' in the model name string."
        )
    save_path = f"{save_dir}/grpo_{size_tag}_{pct}pct"

    model, tokenizer = load_base_model(model_name)
    train_slice, val_slice = load_gsm8k_splits(data_fraction, seed)

    print(f"Model: {size_tag}  |  Training examples: {len(train_slice)} ({pct}%)")
    print(f"Validation examples: {len(val_slice)}")

    run_sft_warmup(model, tokenizer, warmup_path, save_path)

    grpo_dataset = prepare_grpo_dataset(train_slice)
    callback = EarlyStoppingCallback(model, tokenizer, val_slice, save_path)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=grpo_dataset,
        reward_funcs=[compute_reward],
        args=GRPOConfig(
            num_generations=4,
            max_prompt_length=512,
            max_completion_length=512,
            temperature=0.7,
            beta=0.04,
            learning_rate=2e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=max_epochs,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir=save_path + "_grpo_tmp",
            report_to="none",
            seed=seed,
        ),
        callbacks=[callback],
    )

    trainer.train()

    save_training_metrics(
        trainer,
        callback,
        save_path,
        meta={
            "size_tag": size_tag,
            "pct": pct,
            "data_fraction": data_fraction,
            "seed": seed,
            "max_epochs": max_epochs,
            "model_name": model_name,
        },
    )

    print(f"GRPO complete. Best val accuracy: {callback.best_accuracy:.4f}")
    print(f"Best checkpoint: {save_path}_best")
    return model, tokenizer, callback.best_accuracy