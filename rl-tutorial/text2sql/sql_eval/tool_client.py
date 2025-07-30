#!/usr/bin/env python3
"""
SQL tool client module

This module is responsible for:
1. Encapsulating SQL tool call logic
2. Supporting concurrent execution
3. Result formatting and error handling
4. Execution logging
"""

import sqlite3
import json
import logging
import multiprocessing as mp
import re
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
import pandas as pd

try:
    from func_timeout import func_timeout, FunctionTimedOut
except ImportError:
    # 如果没有func_timeout，提供一个简单的替代实现
    def func_timeout(timeout, func, *args, **kwargs):
        return func(*args, **kwargs)
    
    class FunctionTimedOut(Exception):
        pass

logger = logging.getLogger(__name__)


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


def execute_sql_with_timeout(data_idx: int, db_file: str, sql: str, timeout: int = 30, output_str: str = "") -> Tuple:
    """
    Execute SQL with timeout protection.
    
    Args:
        data_idx: Index for tracking the query
        db_file: Path to the SQLite database file
        sql: SQL query to execute
        timeout: Timeout in seconds
        output_str: Additional output string to include in result
    
    Returns:
        Tuple containing execution results and metadata
    """
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        raise
    except FunctionTimedOut:
        logger.warning(f"SQL execution timeout for data_idx: {data_idx}")
        res = (data_idx, db_file, sql, "SQL execution timeout", 0)
        # data_idx, db_file, sql, execution_res, 1
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        res = (data_idx, db_file, sql, f"Error executing SQL: {e}", 0)

    # Append the output to the tuple
    if isinstance(res, tuple):
        res = res + (output_str,)
        
    return res


class SQLToolClient:
    """SQL tool client that encapsulates SQL execution logic"""
    
    def __init__(self, db_root_path: str, timeout: int = 30, 
                 max_result_chars: int = 9000, max_result_rows: int = 50):
        """
        Initialize SQL tool client
        
        Args:
            db_root_path: Database root path
            timeout: SQL execution timeout
            max_result_chars: Maximum result characters
            max_result_rows: Maximum result rows
        """
        self.db_root_path = Path(db_root_path)
        self.timeout = timeout
        self.max_result_chars = max_result_chars
        self.max_result_rows = max_result_rows
        
        if not self.db_root_path.exists():
            raise ValueError(f"Database root path does not exist: {db_root_path}")
    
    def get_database_file_path(self, db_id: str, data_source: str) -> str:
        """
        Get the full path to a database file based on the data source.
        
        Args:
            db_id: Database identifier
            data_source: Source dataset ('synsql', 'spider', 'bird')
            
        Returns:
            Full path to the database file
            
        Raises:
            NotImplementedError: If data_source is not supported
            FileNotFoundError: If database file doesn't exist
        """
        if data_source == 'synsql':
            db_file = self.db_root_path / "SynSQL-2.5M" / "databases" / db_id / f"{db_id}.sqlite"
        elif data_source == 'spider':
            db_file = self.db_root_path / "test_database" / db_id / f"{db_id}.sqlite"
        elif data_source == 'bird':
            db_file = self.db_root_path / "bird" / "train" / "train_databases" / db_id / f"{db_id}.sqlite"
        else:
            raise NotImplementedError(f"Data source '{data_source}' is not supported")
        
        if not db_file.exists():
            raise FileNotFoundError(f"Database file not found: {db_file}")
        
        return str(db_file)
    
    def postprocess_sql_result(self, raw_result: Any) -> str:
        """
        Postprocess SQL execution result to prevent overly long responses.
        
        This method processes the raw SQL execution result and formats it appropriately
        to avoid token length issues. It handles different result types and applies
        truncation when necessary.
        
        Args:
            raw_result: The raw result from SQL execution, can be various types
                       (frozenset, list, dict, etc.)
                       
        Returns:
            Formatted string representation of the result, truncated if necessary
            
        Raises:
            Exception: If result processing fails
        """
        try:
            if raw_result is None:
                return "No results returned"
            
            # Handle frozenset results (convert to DataFrame) - primary case for SQL results
            if isinstance(raw_result, frozenset):
                if not raw_result:
                    return "Empty result set"
                    
                # Convert frozenset to pandas DataFrame
                df = pd.DataFrame(raw_result)
                res_str = df.to_string(index=False)
                
                # Check if result is too long and apply truncation if needed
                if len(res_str) > self.max_result_chars:
                    logger.info(f"SQL result too long ({len(res_str)} chars), truncating to {self.max_result_rows} rows")
                    
                    # Truncate to specified number of rows
                    truncated_df = df.head(self.max_result_rows)
                    res_str = truncated_df.to_string(index=False)
                    
                    # Add truncation notice - format consistent with reference implementation
                    res_str = f"Truncated to {self.max_result_rows} rows since returned response too long: {res_str}"
                    
                    # Final safety check to ensure result is not still too long
                    if len(res_str) > self.max_result_chars:
                        # If still too long, apply character-level truncation
                        res_str = res_str[:self.max_result_chars] + "... (further truncated)"
                
                return res_str
            
            # Handle list results - convert to DataFrame if possible
            elif isinstance(raw_result, list):
                if not raw_result:
                    return "Empty result list"
                
                # Try to convert list to DataFrame
                try:
                    df = pd.DataFrame(raw_result)
                    res_str = df.to_string(index=False)
                    
                    # Apply same truncation logic as frozenset
                    if len(res_str) > self.max_result_chars:
                        logger.info(f"SQL result too long ({len(res_str)} chars), truncating to {self.max_result_rows} rows")
                        
                        truncated_df = df.head(self.max_result_rows)
                        res_str = truncated_df.to_string(index=False)
                        res_str = f"Truncated to {self.max_result_rows} rows since returned response too long: {res_str}"
                        
                        # Final safety check
                        if len(res_str) > self.max_result_chars:
                            res_str = res_str[:self.max_result_chars] + "... (further truncated)"
                    
                    return res_str
                except Exception as e:
                    logger.warning(f"Failed to convert list to DataFrame: {e}, falling back to string representation")
                    # If DataFrame conversion fails, handle as string
                    res_str = str(raw_result)
                    if len(res_str) > self.max_result_chars:
                        res_str = res_str[:self.max_result_chars] + "... (truncated)"
                    return res_str
            
            # Handle dict results - convert to JSON format
            elif isinstance(raw_result, dict):
                res_str = json.dumps(raw_result, indent=2, ensure_ascii=False)
                if len(res_str) > self.max_result_chars:
                    logger.info(f"SQL result too long ({len(res_str)} chars), truncating")
                    res_str = res_str[:self.max_result_chars] + "... (truncated)"
                return res_str
            
            # Handle other types (string, number, etc.)
            else:
                res_str = str(raw_result)
                if len(res_str) > self.max_result_chars:
                    logger.info(f"SQL result too long ({len(res_str)} chars), truncating")
                    res_str = res_str[:self.max_result_chars] + "... (truncated)"
                return res_str
                
        except Exception as e:
            logger.error(f"Error in postprocess_sql_result: {e}")
            return f"Error processing SQL result: {str(e)}"
    
    def execute_sql(self, sql_query: str, db_id: str, data_source: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Execute SQL query using the same logic as sql_tool.py
        
        Args:
            sql_query: SQL query string
            db_id: Database ID
            data_source: Data source type
            
        Returns:
            (response_text, success, metrics)
        """
        logger.info(f"[SQLToolClient] Executing SQL: {sql_query[:100]}...")
        
        # Validate parameters
        if not sql_query or not isinstance(sql_query, str):
            error_msg = "SQL query must be a non-empty string"
            return json.dumps({"result": error_msg}), False, {"error": error_msg}
        
        # Validate SQL syntax
        is_valid, error_msg = validate_sql_syntax(sql_query)
        if not is_valid:
            error_response = json.dumps({"result": f"Invalid SQL syntax: {error_msg}"})
            return error_response, False, {"error": error_msg}
        
        try:
            # Get database file path
            db_file = self.get_database_file_path(db_id, data_source)
            
            # Execute SQL query with timeout using the same logic as sql_execution_utils
            start_time = perf_counter()
            execution_result = execute_sql_with_timeout(0, db_file, sql_query, self.timeout, "")
            execution_time = perf_counter() - start_time
            
            # Parse execution result - format: (data_idx, db_file, sql, execution_res, success, output_str)
            data_idx, db_file_result, sql_result, execution_res, success, output_str = execution_result
            
            # Convert success flag to boolean
            success = bool(success)
            
            if success:
                # SQL execution successful, process query result
                processed_result = self.postprocess_sql_result(execution_res)
            else:
                # SQL execution failed, return error information
                if isinstance(execution_res, str):
                    # execution_res contains error message
                    processed_result = execution_res
                else:
                    # Generic error message
                    processed_result = "SQL execution failed"
            
            response_data = {"result": processed_result}
            response_text = json.dumps(response_data, indent=2, ensure_ascii=False)
            
            # Build execution metrics
            metrics = {
                "sql_query": sql_query,
                "database": db_id,
                "data_source": data_source,
                "successful": success,
                "execution_time": execution_time,
                "error": execution_res if not success and isinstance(execution_res, str) else None
            }
            
            logger.info(f"[SQLToolClient] Executed query, success: {success}")
            return response_text, success, metrics
            
        except Exception as e:
            error_response = json.dumps({"result": f"SQL execution failed: {str(e)}"})
            logger.error(f"[SQLToolClient] Execution failed: {e}")
            return error_response, False, {"error": str(e)}
    
    def execute_sql_batch(self, queries: List[Tuple[str, str, str]], 
                         num_processes: int = 4) -> List[Tuple[str, bool, Dict[str, Any]]]:
        """
        Execute SQL queries in batch using parallel processing
        
        Args:
            queries: List of queries, each element is (sql_query, db_id, data_source)
            num_processes: Number of parallel processes
            
        Returns:
            List of execution results
        """
        if not queries:
            return []
        
        logger.info(f"Executing {len(queries)} SQL queries in parallel with {num_processes} processes")
        
        # Prepare arguments for parallel execution
        args_list = []
        for sql_query, db_id, data_source in queries:
            args_list.append((sql_query, db_id, data_source))
        
        # Use multiprocessing for execution
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_processes) as pool:
                results = pool.starmap(self._execute_single_for_batch, args_list)
            
            logger.info(f"Completed batch SQL execution")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch SQL execution: {e}")
            # Return error result
            error_result = (json.dumps({"result": f"Batch execution error: {str(e)}"}), False, {"error": str(e)})
            return [error_result] * len(queries)
    
    def _execute_single_for_batch(self, sql_query: str, db_id: str, data_source: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Single query execution method for batch execution
        
        This method needs to recreate SQLToolClient instance because instances cannot be shared
        in multiprocessing environment
        """
        try:
            # Recreate client in subprocess
            client = SQLToolClient(
                db_root_path=str(self.db_root_path),
                timeout=self.timeout,
                max_result_chars=self.max_result_chars,
                max_result_rows=self.max_result_rows
            )
            return client.execute_sql(sql_query, db_id, data_source)
        except Exception as e:
            error_response = json.dumps({"result": f"SQL execution failed: {str(e)}"})
            return error_response, False, {"error": str(e)}


def test_sql_tool_client():
    """Test SQL tool client functionality"""
    import tempfile
    import os
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
        temp_db = f.name
    
    try:
        # Create test database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER
            )
        """)
        
        # Insert test data
        cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 25)")
        cursor.execute("INSERT INTO users (name, age) VALUES ('Bob', 30)")
        cursor.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)")
        
        conn.commit()
        conn.close()
        
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        test_db_dir = Path(temp_dir) / "spider" / "database" / "test_db"
        test_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy database file
        import shutil
        shutil.copy2(temp_db, test_db_dir / "test_db.sqlite")
        
        # Test SQL tool client
        client = SQLToolClient(temp_dir)
        
        # Test query
        response, success, metrics = client.execute_sql(
            "SELECT COUNT(*) FROM users",
            "test_db",
            "spider"  # Use spider data source for testing
        )
        
        print(f"Query success: {success}")
        print(f"Response: {response}")
        print(f"Metrics: {metrics}")
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_sql_tool_client() 