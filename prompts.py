SYSTEM_PROMPT = (
    "You are a math reasoning assistant. Solve the problem step by step.\n\n"
    "Rules:\n"
    "1. Wrap all reasoning in <think>...</think> tags.\n"
    "2. Each reasoning step must be its own <step>...</step> tag inside <think>.\n"
    "3. Every <step> must contain the arithmetic expression for that step "
    "using standard operators (+, -, *, /) with numbers (e.g. \"48 / 2 = 24\").\n"
    "4. After </think>, write your final answer as a single integer inside <answer>...</answer> tags.\n"
    "5. No text outside these tags.\n\n"
    "Format:\n"
    "<think><step>expression = result</step><step>expression = result</step></think>\n"
    "<answer>integer</answer>"
)

PRM_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

BASELINE_PROMPT_TEMPLATE = (
    "Solve the math problem. "
    "Write only the final numeric answer on the last line.\n\n{question}"
)

# Qwen2.5-Math-Instruct's officially recommended CoT system prompt.
# The model is finetuned to wrap its final answer in \boxed{}.
MATH_BASELINE_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
