import re


def parse_gsm8k_ops(answer: str) -> tuple[int, list[str | None]]:
    blocks = re.findall(r'<<([^>]+)>>', answer)
    operators = []
    for block in blocks:
        expr = block.split('=')[0].strip()
        expr = re.sub(r'^[+\-]', '', expr)  # strip leading sign from first operand
        op = re.search(r'[+\-*/]', expr)
        operators.append(op.group() if op else None)
    return len(blocks), operators


def normalize_answer(s: str) -> str:
    s = s.replace(',', '').replace('$', '').strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def extract_gsm8k_ground_truth(solution: str) -> str | None:
    match = re.search(r'####\s*([^\n]+)', solution)
    if match:
        return normalize_answer(match.group(1).strip())
    return None


def extract_tagged_answer(text: str) -> str | None:
    match = re.search(r'<answer>\s*([\d,.$]+)\s*</answer>', text)
    if match:
        return normalize_answer(match.group(1))
    return None


def extract_last_integer(text: str) -> str | None:
    nums = re.findall(r'\d[\d,]*', text)
    if nums:
        return normalize_answer(nums[-1])
    return None


def extract_steps(text: str) -> list[str]:
    return re.findall(r'<step>(.*?)</step>', text, re.DOTALL)


def extract_step_operator(step_text: str) -> str | None:
    match = re.search(r'(?<=\d)\s*([+\-*/])\s*(?=\d)', step_text)
    if match:
        return match.group(1)
    return None


def validate_format(text: str) -> bool:
    pattern = r'^<think>(\s*<step>.+?</step>\s*)+</think>\s*<answer>\d+</answer>\s*$'
    return bool(re.match(pattern, text.strip(), re.DOTALL))
