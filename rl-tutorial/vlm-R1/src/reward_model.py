# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import re
from typing import Optional

from openai import OpenAI

PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''



def is_format_correct(completion):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
        return False
    # check if all tags only appear once
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        if completion.count(tag) != 1:
            return False
    
    # check if <think>...</think> is empty
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
    if think_match and think_match.group(1).strip() == "":
        return False
    
    return True

def last_boxed_only_string(string: str):
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    extracted_answer = solution_str
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def rule_based_verify(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def extract_answer_part(response):
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    return ""


def compute_score(data_source, solution_str, ground_truth, extra_info):
    if extra_info is None or "question" not in extra_info or "url" not in extra_info:
        raise ValueError("Extra info is required and must contain 'question' and 'url'")
    
    do_print = False
    if random.randint(0, 512) == 1:  
        do_print = True
    if do_print:
        print(f"Response Case: {solution_str}, Question: {extra_info['question']}, GT: {ground_truth}")

    response = solution_str

    if not is_format_correct(response):
        if do_print:
            print(f"[Invalid Format] Response Case: {response}")
        return {
            "score": -1.0,
            "acc": 0.0,
            "pred": "[INVALID FORMAT]",
        }
    final_answer = extract_answer_part(response)
    if final_answer == "":
        if do_print:
            print(f"[Empty Answer] Response Case: {response}")
        return {
            "score": -1.0,
            "acc": 0.0,
            "pred": "[EMPTY ANSWER]",
        }

    # first, use rule-based verification
    correct, pred = rule_based_verify(final_answer, ground_truth, strict_box_verify=False)
    correct2, pred2 = rule_based_verify(final_answer, ground_truth, strict_box_verify=True)
    if correct or correct2:
        if do_print:
            print(f"[Correct][Rule-based] Answer: {final_answer}, GT: {ground_truth}, Question: {extra_info['question']}")
        return {
            "score": 1.0,
            "acc": 1.0,
            "pred": pred,
        }

    # 如果 ground_truth 和 pred pred2 都是纯数字，且仍旧没有通过 rule-based verification，则返回 -1
    if ground_truth.isdigit() and \
        ((pred is not None and pred.isdigit()) or \
         (pred2 is not None and pred2.isdigit())):
        if do_print:
            print(f"[Incorrect][Rule-based] Answer: {final_answer}, GT: {ground_truth}, Question: {extra_info['question']}")
        return {
            "score": -1.0,
            "acc": 0.0,
            "pred": f"{pred} or {pred2}",
        }
    # extract <answer>...</answer>
    final_answer = final_answer[-400:]

    model = OpenAI(
        base_url=extra_info["url"],
        api_key="EMPTY"
    )

    prompt = PROMPT.format(question=extra_info["question"], output=final_answer, answer=ground_truth)

    try:
        response_obj = model.chat.completions.create(
            model=extra_info["model_name"],
            messages=[
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            temperature=0.1,
            max_tokens=128,
            top_p=0.7,
            timeout=60
        )
        if do_print:
            print(f"[RewardModel] Response: {response_obj.choices[0].message.content}")
        if response_obj.choices[0].message.content == "Correct":
            if do_print:
                print(f"[Correct][RewardModel] Answer: {final_answer}, GT: {ground_truth}, Question: {extra_info['question']}")
            return {
                "score": 1.0,
                "acc": 1.0,
                "pred": final_answer,
            }
    except Exception as e:
        import traceback
        print(f"[RewardModel] Error: {e}")
        print(f"[RewardModel] Error traceback: {traceback.format_exc()}")
        return {
            "score": -1.0,
            "acc": 0.0,
            "pred": "",
        }
    
    # Very unlikely to be correct after the above matches
    if do_print:
        print(f"[Incorrect][RewardModel] Answer: {final_answer}, GT: {ground_truth}, Question: {extra_info['question']}")
    return {
        "score": -1.0,
        "acc": 0.0,
        "pred": "",
    }

# if __name__ == "__main__":
#     print(compute_score("", "1+1=2", "1+1=2", {"question": "1+1=?"}))