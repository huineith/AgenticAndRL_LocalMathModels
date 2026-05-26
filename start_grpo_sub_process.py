import argparse
from grpo_trainer import run_grpo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--warmup_path", default="warmup_data.json")
    p.add_argument("--max_epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _, _, best_acc = run_grpo(
        model_name=args.model,
        data_fraction=args.fraction,
        save_dir=args.save_dir,
        warmup_path=args.warmup_path,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )
    print(f"DONE best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()