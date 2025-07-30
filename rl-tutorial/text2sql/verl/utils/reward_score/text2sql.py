# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import multiprocessing as mp
import random
import re
import sqlite3
import logging
from typing import Optional, List, Tuple, Any, Dict
from pathlib import Path
from time import perf_counter
from func_timeout import func_timeout, FunctionTimedOut

logger = logging.getLogger(__name__)

# Constants for parsing SQL responses
THINK_START, THINK_END = "<think>", "</think>"
TOOL_RESPONSE_START, TOOL_RESPONSE_END = "<tool_response>", "</tool_response>"
ANSWER_START, ANSWER_END = "<answer>", "</answer>"


def extract_sql_solution(solution_str: str) -> Optional[str]:
    """Extract SQL query from the solution string using <answer> tags.
    
    Args:
        solution_str: The complete solution text containing SQL query
        
    Returns:
        The extracted SQL query string or None if extraction fails
    """
    do_print = random.randint(1, 64) == 1

    # # 创建日志目录
    # import os
    # log_dir = Path("/apps/verl_multi-turn/verl/verl/log_data/original_solution_str")
    # os.makedirs(log_dir, exist_ok=True)

    # # 将solution_str写入文件，文件名为当前时间
    # import time
    # from datetime import datetime
    
    # # 获取当前时间并格式化为年月日时分秒
    # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # log_file_path = log_dir / f"solution_{current_time}.txt"
    
    # # 将solution_str写入文件
    # try:
    #     with open(log_file_path, "w", encoding="utf-8") as f:
    #         f.write(solution_str)
    #     if do_print:
    #         print(f"已将solution_str保存到文件: {log_file_path}")
    # except Exception as e:
    #     logger.error(f"保存solution_str到文件时出错: {e}")

    
    if do_print:
        print("--------------------------------")
        print(f"Original solution string: {solution_str[:200]}...")
    
    # Extract from <answer> tags
    if solution_str.count(ANSWER_START) == 1 and solution_str.count(ANSWER_END) == 1:
        try:
            pre_answer, tail = solution_str.split(ANSWER_START, 1)
            answer_text, _ = tail.split(ANSWER_END, 1)
            # Check if there are forbidden tags inside answer
            if not re.search(r"</?(think|tool_call|tool_response)\b", answer_text, re.I):
                # Verify that there are thoughts before answer
                # thoughts = re.findall(r"<think>(.*?)</think>", solution_str, re.S)
                # if thoughts:
                extracted_sql = answer_text.strip()
                if do_print:
                    print(f"Extracted SQL from <answer> tags: {extracted_sql}")
                return extracted_sql
        except Exception as e:
            if do_print:
                print(f"Error extracting from <answer> tags: {e}")
        
    return None


def execute_sql_for_scoring(db_file: str, sql: str, timeout: int = 30) -> Tuple[Optional[frozenset], bool]:
    """Execute SQL query for scoring purposes.
    
    Args:
        db_file: Path to the SQLite database file
        sql: SQL query to execute
        timeout: Timeout for execution in seconds
        
    Returns:
        Tuple of (execution_result, success_flag)
    """
    def _execute():
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            conn.execute("BEGIN TRANSACTION;")
            cursor.execute(sql)
            result = frozenset(cursor.fetchall())
            conn.rollback()
            return result, True
        except Exception as e:
            logger.debug(f"SQL execution error: {e}")
            return None, False
        finally:
            if conn:
                conn.close()
    
    try:
        return func_timeout(timeout, _execute)
    except FunctionTimedOut:
        logger.debug(f"SQL execution timeout: {sql[:100]}...")
        return None, False
    except Exception as e:
        logger.debug(f"SQL execution error: {e}")
        return None, False


def sql_results_match(pred_result: Optional[frozenset], gt_result: Optional[frozenset]) -> bool:
    """Check if predicted SQL result matches ground truth result.
    
    Args:
        pred_result: Result from predicted SQL query
        gt_result: Result from ground truth SQL query
        
    Returns:
        True if results match, False otherwise
    """
    if pred_result is None or gt_result is None:
        return False
    return pred_result == gt_result


def compute_score(solution_str: str, ground_truth: Dict[str, Any], 
                 method: str = "strict", format_score: float = -1.0, 
                 score: float = 1.0, timeout: int = 30) -> float:
    """Compute score for text2sql task.
    
    This function follows the same signature pattern as other compute_score functions
    in the framework, while implementing text2sql specific logic.
    
    Args:
        solution_str: The solution text containing SQL query
        ground_truth: Dictionary containing:
            - 'sql': Ground truth SQL query
            - 'db_file': Path to database file
            - 'data_source': Data source type (synsql, spider, bird)
        method: Scoring method ('strict' or 'flexible')
        format_score: Score given for correct format but wrong answer
        score: Score given for correct answer
        timeout: SQL execution timeout in seconds
        
    Returns:
        Float score: format_score for format errors, 0.0 for wrong answers, 
                    score for correct answers
    """
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print("================================")
        print(f"Computing text2sql score...")
        print(f"Ground truth SQL: {ground_truth.get('sql', '')}")
        print(f"Database file: {ground_truth.get('db_file', '')}")
        print(f"Solution string: {solution_str[:300]}...")
    
    # Extract SQL from solution
    predicted_sql = extract_sql_solution(solution_str)
    
    if predicted_sql is None:
        if do_print:
            print("❌ Failed to extract SQL - format error")
        return format_score
    
    # Get database file and ground truth SQL
    db_file = ground_truth.get('db_file')
    gt_sql = ground_truth.get('sql')
    
    if not db_file or not gt_sql:
        if do_print:
            print("❌ Missing database file or ground truth SQL")
        return 0.0
    
    # Check if database file exists
    if not Path(db_file).exists():
        if do_print:
            print(f"❌ Database file not found: {db_file}")
        return 0.0
    
    # Execute both predicted and ground truth SQL
    pred_result, pred_success = execute_sql_for_scoring(db_file, predicted_sql, timeout)
    gt_result, gt_success = execute_sql_for_scoring(db_file, gt_sql, timeout)
    
    # # 将评分过程中的关键变量保存到文件中
    # import os
    # import json
    # import uuid
    # import time

    # # 创建日志目录
    # log_dir = Path("/apps/verl_multi-turn/verl/verl/log_data/reward_result")
    # os.makedirs(log_dir, exist_ok=True)

    # # 生成唯一的文件名
    # unique_id = str(uuid.uuid4())
    # timestamp = int(time.time())
    # log_file_path = log_dir / f"sql_reward_{timestamp}_{unique_id}.json"

    # # 准备要保存的数据 - 将frozenset转换为list以便JSON序列化
    # def convert_frozenset_to_list(obj):
    #     """将frozenset转换为list，便于JSON序列化"""
    #     if obj is None:
    #         return None
    #     elif isinstance(obj, frozenset):
    #         return list(obj)
    #     else:
    #         return obj
    
    # log_data = {
    #     "predicted_sql": predicted_sql,
    #     "pred_result": convert_frozenset_to_list(pred_result),
    #     "pred_success": pred_success,
    #     "gt_sql": gt_sql,
    #     "gt_result": convert_frozenset_to_list(gt_result),
    #     "gt_success": gt_success,
    #     "timestamp": timestamp
    # }

    # if pred_success:
    #     # 将数据保存到文件
    #     with open(log_file_path, "w", encoding="utf-8") as f:
    #         json.dump(log_data, f, ensure_ascii=False, indent=2)
    #         print(f"✅ 已将评分数据保存至: {log_file_path}")

    if do_print:
        print(f"Predicted SQL: {predicted_sql}")
        print(f"Predicted execution success: {pred_success}")
        print(f"Ground truth execution success: {gt_success}")
    
    # If ground truth SQL fails, there might be an issue with the data
    if not gt_success:
        if do_print:
            print("⚠️ Ground truth SQL execution failed")
        return 0.0
    
    # If predicted SQL fails to execute, it's incorrect
    if not pred_success:
        if do_print:
            print("❌ Predicted SQL execution failed")
        return 0.0
    
    # Compare results
    results_match = sql_results_match(pred_result, gt_result)
    
    if do_print:
        print(f"Results match: {results_match}")
        if results_match:
            print("✅ Correct answer!")
        else:
            print("❌ Wrong answer")
            print(f"Predicted result: {pred_result}")
            print(f"Ground truth result: {gt_result}")
    
    
    
    return score if results_match else 0.0


def compute_score_batch(solution_strs: List[str], ground_truths: List[Dict[str, Any]], 
                       method: str = "strict", format_score: float = -1.0, 
                       score: float = 1.0, timeout: int = 30, num_cpus: int = 32) -> List[float]:
    """Compute scores for a batch of text2sql problems in parallel.
    
    This function provides batch processing similar to the original 
    calculate_reward_parallel function but adapted for the compute_score interface.
    
    Args:
        solution_strs: List of solution strings
        ground_truths: List of ground truth dictionaries
        method: Scoring method
        format_score: Score for format errors
        score: Score for correct answers
        timeout: SQL execution timeout
        num_cpus: Number of CPU cores for parallel processing
        
    Returns:
        List of computed scores
    """
    if len(solution_strs) != len(ground_truths):
        raise ValueError("Number of solutions must match number of ground truths")
    
    start_time = perf_counter()
    logger.info(f"Computing text2sql scores for {len(solution_strs)} problems in parallel")
    
    # Prepare arguments for parallel processing
    args_list = [
        (solution_str, gt, method, format_score, score, timeout)
        for solution_str, gt in zip(solution_strs, ground_truths)
    ]
    
    # Use multiprocessing for parallel execution
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_cpus) as pool:
        scores = pool.starmap(compute_score, args_list)
    
    end_time = perf_counter()
    logger.info(f"Completed text2sql scoring in {end_time - start_time:.2f} seconds")
    
    return scores


# Legacy function for backward compatibility with original text2sql code
def sql_compute_score(completions: List[str], references: List[str], 
                     db_files: List[str], tasks: List[str], 
                     questions: Optional[List[str]] = None, 
                     n_agent: Optional[int] = None, 
                     log_dir: Optional[str] = None) -> List[float]:
    """Legacy function compatible with original text2sql interface.
    
    This function maintains compatibility with the original sql_compute_score
    while using the new compute_score implementation internally.
    
    Args:
        completions: List of model completions
        references: List of ground truth SQL queries
        db_files: List of database file paths
        tasks: List of task types (should all be 'synsql')
        questions: Optional list of questions
        n_agent: Optional agent number for logging
        log_dir: Optional directory for logging
        
    Returns:
        List of computed scores
    """
    if len(completions) != len(references) or len(completions) != len(db_files):
        raise ValueError("Length of completions, references, and db_files must match")
    
    # Prepare ground truth dictionaries
    ground_truths = []
    for ref_sql, db_file, task in zip(references, db_files, tasks):
        ground_truths.append({
            'sql': ref_sql,
            'db_file': db_file,
            'data_source': task
        })
    
    # Use batch compute score
    try:
        scores = compute_score_batch(
            solution_strs=completions,
            ground_truths=ground_truths,
            method="strict",
            format_score=-1.0,
            score=1.0,
            timeout=30,
            num_cpus=32
        )
        return scores
    except Exception as e:
        logger.error(f"Error in sql_compute_score: {e}")
        return [0.0] * len(completions) 



if __name__ == "__main__":
    filename = "/apps/verl_multi-turn/verl/verl/log_data/original_solution_str/solution_20250701044807.txt"
    solution_str = open(filename, "r", encoding="utf-8").read()
    print(solution_str)
    print(extract_sql_solution(solution_str))