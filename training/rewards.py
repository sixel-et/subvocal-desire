"""Reward functions for desire detection GRPO training.

Each reward function follows TRL's GRPOTrainer interface:
    def reward_fn(completions, <dataset_columns>, **kwargs) -> list[float]

Where completions is a list of model outputs and additional dataset columns
are passed as keyword arguments.
"""

import re


def get_completion_text(completion) -> str:
    """Extract text from a TRL completion (handles strings, dicts, message lists)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", "")
    if isinstance(completion, list) and len(completion) > 0:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)
    return str(completion)


def extract_answer(text: str):
    """Extract numerical answer from model output. Returns int or None.

    Handles patterns like:
        "Answer: 49"
        "The result of 7 * 7 is 49."
        "7 * 7 = 49"
        "\\[ 601 \\times 304 = 182544 \\]"
        "216,592"  (comma-separated, as the last big number)
    """
    # Try "Answer: <number>" format first (most explicit)
    match = re.search(r'[Aa]nswer:\s*([+-]?\d[\d,]*)', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Try "is <number>" — model says "The result of X * Y is Z"
    match = re.search(r'\bis\s+([+-]?\d[\d,]*)', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Try "= <number>" — model says "X * Y = Z" or LaTeX "\times ... = Z"
    match = re.search(r'=\s*([+-]?\d[\d,]*)', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Fallback: last standalone number (not first — first is often from problem restatement)
    numbers = re.findall(r'\b\d[\d,]*\b', text)
    if numbers:
        try:
            return int(numbers[-1].replace(',', ''))
        except ValueError:
            pass

    return None


def extract_confidence(text: str):
    """Extract confidence score (0-100) from model output. Returns int or None.

    Handles patterns the model actually uses:
        "Confidence: 85"                    (structured)
        "My confidence is 100%"             (prose with %)
        "confidence level of 90"            (prose without %)
        "I'm 95% confident"                 (inline %)
        "score: 85"                         (confidence score)
        "I'm sorry, I can't calculate"      (refusal → 0)

    Rejects echoed template patterns like "0-100" or "<0-100>".
    """
    text_lower = text.lower()

    # Refusal = confidence 0
    if "i'm sorry" in text_lower or "i can't" in text_lower or "i apologize" in text_lower:
        return 0

    # Reject template echo: "Confidence: 0-100" or "<0-100>"
    match = re.search(r'[Cc]onfidence:\s*<?(\d+)(?:\s*[-–]|>)', text)
    if match:
        return None

    # "Confidence: N" or "confidence is N" or "confidence of N" or "confidence level ... N"
    match = re.search(r'[Cc]onfidence[\s\w]*?[:is]\s*(\d+)', text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val

    # "N%" anywhere after a confidence-related word
    # Look for percentage near confidence/confident/accuracy/certain
    conf_region = re.search(r'(confident|confidence|accuracy|certain).*?(\d+)%', text_lower)
    if conf_region:
        val = int(conf_region.group(2))
        if 0 <= val <= 100:
            return val

    # Standalone "N%" (common: "My confidence in this answer is 100%.")
    # Use the LAST percentage found (avoids matching problem numbers)
    pct_matches = re.findall(r'(\d+)%', text)
    if pct_matches:
        val = int(pct_matches[-1])
        if 0 <= val <= 100:
            return val

    # "score: N" or "score of N"
    match = re.search(r'score[:\s]+(\d+)', text_lower)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val

    return None


# ---------------------------------------------------------------------------
# Tool request detection
# ---------------------------------------------------------------------------

TOOL_PATTERN = re.compile(
    r'<tool>\s*calculator\s*</tool>\s*<input>\s*([^<]+)\s*</input>',
    re.IGNORECASE
)


def has_tool_request(text: str) -> bool:
    """Check if text contains a tool request."""
    return bool(TOOL_PATTERN.search(text))


def extract_tool_expression(text: str) -> str | None:
    """Extract the arithmetic expression from a tool request."""
    match = TOOL_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def execute_calculator(expression: str) -> int | None:
    """Safely evaluate a simple arithmetic expression."""
    # Only allow digits, spaces, and basic operators
    cleaned = re.sub(r'[^\d\s\+\-\*\/\(\)]', '', expression)
    if not cleaned:
        return None
    try:
        result = eval(cleaned, {"__builtins__": {}}, {})
        return int(result)
    except:
        return None


# ---------------------------------------------------------------------------
# Stage 1b: Tool Use Reward
# ---------------------------------------------------------------------------

def tool_use_reward(completions, answer, **kwargs) -> list[float]:
    """
    Reward for Stage 1b: learning when to request a calculator.

    Reward ordering (the key design constraint):
        R(correct-without-tool) > R(tool-request) > R(wrong-without-tool)

    This ensures GRPO nudges toward tool use on problems where the model
    is unreliably correct, because expected reward of tool-requesting
    exceeds expected reward of sometimes-right-sometimes-wrong answering.

    Specific values:
        correct-without-tool:     1.0  (ideal: model knew it and got it right)
        tool-request + correct:   0.8  (good: model asked for help, used it well)
        tool-request + wrong:     0.4  (ok: asked for help but fumbled the answer)
        wrong-without-tool:       0.0  (bad: should have asked for help)
        unparseable:              0.1  (minimal: learn to produce valid output)

    The tool-request rewards (0.8, 0.4) are set so that even when the model
    uses the tool imperfectly, tool-requesting beats unreliable answering
    on problems where accuracy < 80%.

    Args:
        completions: list of model outputs
        answer: list of ground truth answers (ints)
    """
    rewards = []

    for completion, ground_truth in zip(completions, answer):
        text = get_completion_text(completion)
        ground_truth = int(ground_truth)

        if has_tool_request(text):
            # Model requested a tool
            expr = extract_tool_expression(text)
            if expr:
                tool_result = execute_calculator(expr)
                if tool_result is None:
                    # Tool expression didn't execute (e.g., "bad", malformed math)
                    rewards.append(0.3)
                else:
                    # Tool executed successfully
                    extracted = extract_answer(text)
                    if extracted is not None and extracted == ground_truth:
                        rewards.append(0.8)  # Tool request + correct answer
                    elif tool_result == ground_truth:
                        # Tool gave right answer but model fumbled it
                        rewards.append(0.5)
                    else:
                        rewards.append(0.4)  # Tool request + wrong answer
            else:
                rewards.append(0.3)  # Malformed tool request (no expression extracted)
        else:
            # Model answered directly (no tool)
            extracted = extract_answer(text)
            if extracted is None:
                rewards.append(0.1)  # Unparseable
            elif extracted == ground_truth:
                rewards.append(1.0)  # Correct without tool (best)
            else:
                rewards.append(0.0)  # Wrong without tool (worst)

    return rewards


# ---------------------------------------------------------------------------
# Stage 1a: Calibration Reward
# ---------------------------------------------------------------------------

def calibration_reward(completions, answer, **kwargs) -> list[float]:
    """
    Reward for Stage 1a: calibrated confidence using Brier scoring.

    Uses a continuous proper scoring rule so GRPO gets gradient signal from
    any variation in confidence across completions (discrete buckets caused
    reward_std=0 on 80-100% of batches → zero gradient).

    Brier score: reward = 1 - (confidence - is_correct)^2
        Correct + conf 95  → 1 - 0.0025 = 0.997
        Correct + conf 50  → 1 - 0.25   = 0.75
        Wrong   + conf 10  → 1 - 0.01   = 0.99
        Wrong   + conf 90  → 1 - 0.81   = 0.19
        Wrong   + conf 50  → 1 - 0.25   = 0.75

    Incentivizes decisive calibration: always saying 50% gives 0.75,
    but proper calibration (100% on easy, 10% on hard) gives ~0.99.

    Args:
        completions: list of model outputs
        answer: list of ground truth answers (ints)
    """
    rewards = []

    for completion, ground_truth in zip(completions, answer):
        text = get_completion_text(completion)

        extracted = extract_answer(text)
        confidence = extract_confidence(text)

        if extracted is None and confidence is None:
            rewards.append(0.05)  # Totally unparseable
            continue

        if extracted is None or confidence is None:
            rewards.append(0.1)  # Partially parseable
            continue

        is_correct = float(extracted == int(ground_truth))
        conf = confidence / 100.0

        # Brier score: proper scoring rule, continuous
        brier = 1.0 - (conf - is_correct) ** 2
        rewards.append(brier)

    return rewards
