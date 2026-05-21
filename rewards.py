from utils import (
    validate_format,
    extract_tagged_answer,
    extract_gsm8k_ground_truth,
    extract_steps,
    extract_step_operator,
    parse_gsm8k_ops,
)


def r_answer(completion: str, solution: str) -> float:
    predicted = extract_tagged_answer(completion)
    expected = extract_gsm8k_ground_truth(solution)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def r_steps(completion: str, solution: str) -> float:
    expected_count, _ = parse_gsm8k_ops(solution)
    if expected_count == 0:
        return 0.0
    actual_count = len(extract_steps(completion))
    return 1.0 if actual_count == expected_count else 0.0


def r_operators(completion: str, solution: str) -> float:
    expected_count, expected_ops = parse_gsm8k_ops(solution)
    actual_steps = extract_steps(completion)
    if len(actual_steps) != expected_count or expected_count == 0:
        return 0.0
    correct = sum(
        1 for step, expected_op in zip(actual_steps, expected_ops)
        if extract_step_operator(step) == expected_op
    )
    return correct / expected_count


# Single reward function for TRL GRPOTrainer.
# `solution` is the raw GSM8K answer field containing <<>> annotations and #### N.
def compute_reward(completions: list[str], solution: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, sol in zip(completions, solution):
        if not validate_format(completion):
            rewards.append(0.0)
            continue
        total = r_answer(completion, sol) + r_steps(completion, sol) + r_operators(completion, sol)
        rewards.append(total)
    return rewards
