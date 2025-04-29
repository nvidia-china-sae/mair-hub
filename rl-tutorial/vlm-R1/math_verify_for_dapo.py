try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    import random
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = -1.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, pred = verify_func([ground_truth_boxed], [solution_str])
        # random print
        if random.randint(0, 512) == 1:
            print(f"ret_score: {ret_score}, pred: {pred}")
    except Exception as e:
        print(f"reward -1.0 for skip the example: {e=}, {ground_truth_boxed=}, {solution_str=}")
        ret_score = -1.0
        pred = solution_str
        
    # Fix the condition check, use a safer way to verify
    if not isinstance(pred, (list, tuple)) or len(pred) == 0 or not isinstance(pred[0], (list, tuple)) or len(pred[0]) < 1:
        print(f"Parser Error: {ret_score=}, {pred=}, {ground_truth_boxed=}, {solution_str=}")
        
        return {
            "score": -1.0,
            "acc": False,
            "pred": "parser_error",
        }
    return post_process(ret_score, pred[0][0])

def post_process(score, pred):
    acc = score == 1.0
    return {
        "score": score,
        "acc": acc,
        "pred": pred,
    }



