# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    # Extract database information from extra_info
    db_file = None
    data_source_type = data_source
    
    if extra_info:
        # Try to get database info from tools_kwargs
        tools_kwargs = extra_info.get("tools_kwargs", {})
        execute_sql_kwargs = tools_kwargs.get("execute_sql", {})
        create_kwargs = execute_sql_kwargs.get("create_kwargs", {})

        if create_kwargs:
            db_id = create_kwargs.get("db_id")
            original_data = create_kwargs.get("original_data", create_kwargs.get("data_source", data_source))
            
            if db_id and original_data:
                # 首先尝试从create_kwargs中获取db_root_path（新的方法）
                db_root_path = create_kwargs.get("db_root_path")
                                    
                # Construct database file path based on data source
                if original_data == 'synsql':
                    db_file = str(Path(db_root_path) / "SynSQL-2.5M" / "databases" / db_id / f"{db_id}.sqlite")
                elif original_data == 'spider':
                    db_file = str(Path(db_root_path) / "spider" / "database" / db_id / f"{db_id}.sqlite")
                elif original_data == 'bird':
                    db_file = str(Path(db_root_path) / "bird" / "train" / "train_databases" / db_id / f"{db_id}.sqlite")
                
                data_source_type = original_data
        
        # Fallback: try to get db_file directly
        if not db_file:
            db_file = extra_info.get('db_file')
    
    return compute_sql_score(solution_str, ground_truth, db_file)

    

def compute_sql_score(solution_str: str, 
                 gt_sql: str, 
                 db_file: str,
                 method: str = "strict", 
                 format_score: float = -1.0, 
                 score: float = 1.0, 
                 timeout: int = 30) -> float:
    """Compute score for text2sql task.
    
    This function follows the same signature pattern as other compute_score functions
    in the framework, while implementing text2sql specific logic.
    
    Args:
        solution_str: The solution text containing SQL query
        gt_sql: the ground truth answer for comparison.
        db_file: Path to the SQLite database file
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
        print(f"Ground truth SQL: {gt_sql}")
        print(f"Database file: {db_file}")
        print(f"Solution string: {solution_str[:300]}...")
    
    # Extract SQL from solution
    predicted_sql = extract_sql_solution(solution_str)
    
    if predicted_sql is None:
        if do_print:
            print("❌ Failed to extract SQL - format error")
        return format_score
    
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


if __name__ == "__main__":
    filename = "/apps/verl/log_data/original_solution_str/solution_20250701044807.txt"
    solution_str = open(filename, "r", encoding="utf-8").read()
    print(solution_str)
    print(extract_sql_solution(solution_str))