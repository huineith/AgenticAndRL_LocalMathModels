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


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the answer from Qwen2.5-Math-Instruct output, which wraps the final
    answer in \\boxed{...}. Handles nested braces by matching them manually.
    Falls back to the last integer in the text if no \\boxed{} is found.
    """
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return extract_last_integer(text)

    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return extract_last_integer(text)

    content = text[start:i - 1].strip()
    # \boxed{} kan innehålla LaTeX som \frac, \%, $, {,} (LaTeX-grupperat komma), etc.
    # Strippa LaTeX-grupperade kommatecken {,} → , och dollartecken före num-match.
    cleaned = content.replace('{,}', ',').replace('$', '').replace('\\,', '')
    m = re.search(r'-?\d[\d,]*(?:\.\d+)?', cleaned)
    if m:
        return normalize_answer(m.group())
    return extract_last_integer(text)


def _extract_boxed_only(text: str) -> str | None:
    """Som extract_boxed_answer men UTAN last-integer-fallback.
    Returnerar None om \\boxed{} saknas eller är tomt. Används för H0
    där vi vill kunna skilja 'inget format alls' från 'fel format'."""
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    content = text[start:i - 1].strip()
    cleaned = content.replace('{,}', ',').replace('$', '').replace('\\,', '')
    m = re.search(r'-?\d[\d,]*(?:\.\d+)?', cleaned)
    return normalize_answer(m.group()) if m else None


def extract_h0_answer(text: str) -> str | None:
    """
    H0-extractor: försök <answer>-taggen först, sedan \\boxed{}.
    Använder INTE last-integer-fallback — vi vill mäta hur ofta otränade
    modeller producerar *något* parsable format, inte plocka random siffror.
    """
    tagged = extract_tagged_answer(text)
    if tagged is not None:
        return tagged
    return _extract_boxed_only(text)


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
