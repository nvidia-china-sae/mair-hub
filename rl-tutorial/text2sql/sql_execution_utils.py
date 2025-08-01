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

import sqlite3
import multiprocessing as mp
import json
import re
import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from time import perf_counter
from func_timeout import func_timeout, FunctionTimedOut

logger = logging.getLogger(__name__)

# Constants for parsing SQL responses - CORRECTED TOKENS
THINK_START, THINK_END = "<think>", "</think>"
ANSWER_START, ANSWER_END = "<answer>", "</answer>"


def execute_sql(data_idx: int, db_file: str, sql: str) -> Tuple[int, str, str, Optional[frozenset], int]:
    """
    Execute a SQL query on a SQLite database.
    
    Args:
        data_idx: Index for tracking the query
        db_file: Path to the SQLite database file
        sql: SQL query to execute
    
    Returns:
        Tuple of (data_idx, db_file, sql, execution_result, success_flag)
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = frozenset(cursor.fetchall())
        conn.rollback()
        return data_idx, db_file, sql, execution_res, 1
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        return data_idx, db_file, sql, f"Error executing SQL: {e}", 0
    finally:
        if conn:
            conn.close()


# def execute_sql_with_timeout(data_idx: int, db_file: str, sql: str, timeout: int = 30, output_str: str = "") -> Tuple:
#     """
#     Execute SQL with timeout protection.
    
#     Args:
#         data_idx: Index for tracking the query
#         db_file: Path to the SQLite database file
#         sql: SQL query to execute
#         timeout: Timeout in seconds
#         output_str: Additional output string to include in result
    
#     Returns:
#         Tuple containing execution results and metadata
#     """
#     try:
#         res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
#     except KeyboardInterrupt:
#         raise
#     except FunctionTimedOut:
#         logger.warning(f"SQL execution timeout for data_idx: {data_idx}")
#         res = (data_idx, db_file, sql, "SQL execution timeout", 0)
#         # data_idx, db_file, sql, execution_res, 1
#     except Exception as e:
#         logger.error(f"Error executing SQL: {e}")
#         res = (data_idx, db_file, sql, f"Error executing SQL: {e}", 0)

#     # Append the output to the tuple
#     if isinstance(res, tuple):
#         res = res + (output_str,)
        
#     return res


def verify_format_and_extract(output: str) -> Tuple[bool, Optional[List[str]], Optional[str], Optional[str]]:
    """
    Verify the format of the output and extract SQL query.
    Updated to use correct token format.
    
    Args:
        output: The model output string
        
    Returns:
        Tuple of (is_valid, thoughts, answer_text, extracted_sql)
    """
    if output.count(ANSWER_START) != 1 or output.count(ANSWER_END) != 1:
        return False, None, None, None

    pre_answer, tail = output.split(ANSWER_START, 1)
    answer_text, _ = tail.split(ANSWER_END, 1)

    if re.search(r"</?(think|tool_call|tool_response)\b", answer_text, re.I):
        return False, None, None, None

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return False, None, None, None

    for m in re.finditer(r"</tool_response>", pre_answer, re.I):
        rest = pre_answer[m.end():].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None, None

    return True, thoughts, answer_text.strip(), None


def execute_sql_single_direct(
    data_idx: int,
    db_file: str,
    sql: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    直接执行单个SQL查询，不创建新进程。
    
    Args:
        data_idx: Index for tracking the query  
        db_file: Path to the SQLite database file
        sql: SQL query to execute
        timeout: Timeout in seconds
        output_str: Additional output string
        
    Returns:
        Dictionary containing execution results
    """
    try:
        # 使用带超时的执行
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
        data_idx, db_file, sql, execution_res, success = res
    except KeyboardInterrupt:
        raise
    except FunctionTimedOut:
        logger.warning(f"SQL execution timeout for data_idx: {data_idx}")
        data_idx, db_file, sql, execution_res, success = (data_idx, db_file, sql, "SQL execution timeout", 0)
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        data_idx, db_file, sql, execution_res, success = (data_idx, db_file, sql, f"Error executing SQL: {e}", 0)
    
    # 构建结果字典
    result_dict = {
        "index": data_idx,
        "db_file": db_file,
        "sql": sql,
        "execution_result": execution_res,
        "success": bool(success),
        "error": None if success else "Execution failed"
    }
    
    return result_dict 


# def execute_sql_batch_parallel(
#     sql_queries: List[str],
#     db_files: List[str],
#     num_cpus: int = 32,
#     timeout: int = 30,
#     log_dir: Optional[str] = None
# ) -> List[Dict[str, Any]]:
#     """
#     Execute a batch of SQL queries in parallel.
    
#     Args:
#         sql_queries: List of SQL queries to execute
#         db_files: List of database file paths (one for each query)
#         num_cpus: Number of CPU cores to use
#         timeout: Timeout for each SQL execution
#         log_dir: Optional directory to log results
        
#     Returns:
#         List of execution results with metadata
#     """
#     if len(sql_queries) != len(db_files):
#         raise ValueError("Number of SQL queries must match number of database files")
    
#     start_time = perf_counter()
#     logger.info(f"Starting parallel execution of {len(sql_queries)} SQL queries")
    
#     # Prepare tasks for parallel execution
#     tasks = []
#     for i, (sql, db_file) in enumerate(zip(sql_queries, db_files)):
#         tasks.append((i, db_file, sql, timeout, ""))
    
#     # Execute in parallel using multiprocessing
#     ctx = mp.get_context("spawn")
#     with ctx.Pool(processes=num_cpus) as pool:
#         results = pool.starmap(execute_sql_with_timeout, tasks)
    
#     # Process results
#     execution_results = []
#     for result in results:
#         data_idx, db_file, sql, execution_res, success, output_str = result
        
#         result_dict = {
#             "index": data_idx,
#             "db_file": db_file,
#             "sql": sql,
#             "execution_result": execution_res,
#             "success": bool(success),
#             "output": output_str,
#             "error": None if success else "Execution failed"
#         }
#         execution_results.append(result_dict)
    
#     # Log results if directory is provided
#     if log_dir:
#         log_dir = Path(log_dir)
#         log_dir.mkdir(parents=True, exist_ok=True)
        
#         for i, result in enumerate(execution_results):
#             log_file = log_dir / f"sql_execution_{i}.json"
#             with open(log_file, "w") as f:
#                 json.dump(result, f, default=str, indent=2)
    
#     end_time = perf_counter()
#     logger.info(f"Completed parallel SQL execution in {end_time - start_time:.2f} seconds")
    
#     return execution_results


def get_database_file_path(db_id: str, data_source: str, db_root_path: str) -> str:
    """
    Get the full path to a database file based on the data source.
    
    Args:
        db_id: Database identifier
        data_source: Source dataset ('synsql', 'spider', 'bird')
        db_root_path: Root path containing database directories
        
    Returns:
        Full path to the database file
        
    Raises:
        NotImplementedError: If data_source is not supported
        FileNotFoundError: If database file doesn't exist
    """
    db_root = Path(db_root_path)
    
    if data_source == 'synsql':
        db_file = db_root / "SynSQL-2.5M" / "databases" / db_id / f"{db_id}.sqlite"
    elif data_source == 'spider':
        db_file = db_root / "spider" / "database" / db_id / f"{db_id}.sqlite"
    elif data_source == 'bird':
        db_file = db_root / "bird" / "train" / "train_databases" / db_id / f"{db_id}.sqlite"
    else:
        raise NotImplementedError(f"Data source '{data_source}' is not supported")
    
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_file}")
    
    return str(db_file)


def validate_sql_syntax(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Basic SQL syntax validation.
    
    Args:
        sql: SQL query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    
    # Basic checks for common SQL keywords
    sql_upper = sql.upper().strip()
    if not any(keyword in sql_upper for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP']):
        return False, "No valid SQL command found"
    
    # Check for balanced parentheses
    if sql.count('(') != sql.count(')'):
        return False, "Unbalanced parentheses in SQL query"
    
    return True, None 
