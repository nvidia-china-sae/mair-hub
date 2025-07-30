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

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar, List, Dict, Union
from uuid import uuid4
from pathlib import Path

import ray
import ray.actor
import pandas as pd

from verl.tools.utils.sql_execution_utils import (
    execute_sql_single_direct,
    get_database_file_path,
    validate_sql_syntax,
    
)

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SqlExecutionWorker:
    """Worker for executing SQL operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="sql-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> Optional[T]:
        """Execute function with optional rate limiting.
        
        Returns:
            The result of the function execution, or None if an exception occurred.
            This prevents exception propagation that could cause Ray actor resource leaks.
        """
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing SQL: {e}")
                    # 不重新抛出异常，让上层统一处理
                    return None  # 返回None表示执行失败
        else:
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing SQL: {e}")
                return None


def init_sql_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """Initialize SQL execution pool."""
    if mode == PoolMode.ThreadMode:
        return ray.remote(SqlExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SqlTool(BaseTool):
    """SQL execution tool for executing SQL queries against databases.

    This tool provides SQL execution functionality with rate limiting and concurrent execution
    support through Ray. It supports multiple database formats (SynSQL, Spider, Bird) and
    includes comprehensive error handling and logging.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the SQL tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
        postprocess_sql_result: Postprocess SQL execution results to prevent overly long responses
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SqlTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Executes SQL queries and returns the results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql_query": {
                                "type": "string",
                                "description": "SQL query to be executed"
                            }
                        },
                        "required": ["sql_query"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 60)
        self.rate_limit = config.get("rate_limit", 60)
        self.timeout = config.get("timeout", 30)
        self.num_cpus = config.get("num_cpus", 32)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_sql_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode
        )

        # Database configuration
        self.db_root_path = config.get("db_root_path")
        assert self.db_root_path, "Configuration must include 'db_root_path'"
        if not Path(self.db_root_path).exists():
            raise ValueError(f"Database root path does not exist: {self.db_root_path}")

        # Logging configuration
        self.enable_logging = config.get("enable_logging", False)
        self.log_dir = config.get("log_dir", None)
        
        # Postprocessing configuration
        self.max_result_chars = config.get("max_result_chars", 9000)
        self.max_result_rows = config.get("max_result_rows", 50)

        logger.info(f"Initialized SqlTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            **kwargs: Additional keyword arguments containing database information.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Extract database information from kwargs
        db_id = kwargs.get("db_id")
        data_source = kwargs.get("original_data", kwargs.get("data_source"))
        
        self._instance_dict[instance_id] = {
            "execution_results": [],
            "metrics": {},
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "db_id": db_id,
            "data_source": data_source,
        }
        return instance_id

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate the input parameters."""
        sql_query = parameters.get("sql_query")

        if not sql_query or not isinstance(sql_query, str):
            return False, "sql_query must be a non-empty string"
        
        # Validate SQL syntax
        is_valid, error_msg = validate_sql_syntax(sql_query)
        if not is_valid:
            return False, f"Invalid SQL syntax: {error_msg}"
        
        return True, None

    def get_database_info_from_instance(self, instance_id: str) -> Tuple[str, str]:
        """Get database information from instance data.
        
        Args:
            instance_id: The instance ID
            
        Returns:
            Tuple of (db_id, data_source)
            
        Raises:
            ValueError: If database information is missing
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance_data = self._instance_dict[instance_id]
        db_id = instance_data.get("db_id")
        data_source = instance_data.get("data_source")
        
        if not db_id:
            raise ValueError("Database ID (db_id) not found in instance data")
        if not data_source:
            raise ValueError("Data source not found in instance data")
            
        return db_id, data_source

    def prepare_database_file(self, db_id: str, data_source: str) -> str:
        """Prepare the database file path."""
        try:
            db_file = get_database_file_path(db_id, data_source, self.db_root_path)
            return db_file
        except (NotImplementedError, FileNotFoundError) as e:
            logger.error(f"Error getting database file for {db_id} ({data_source}): {e}")
            raise ValueError(f"Cannot access database {db_id} from {data_source}: {e}")

    def execute_sql_single(
        self,
        instance_id: str,
        sql_query: str,
        db_file: str,
        timeout: int,
        log_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Execute a single SQL query without creating new processes."""
        import json
        from pathlib import Path
        
        # 直接执行SQL，不创建新进程池
        result = execute_sql_single_direct(
            data_idx=0,
            db_file=db_file,
            sql=sql_query,
            timeout=timeout,
        )
        
        # 如果需要记录日志
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / "sql_execution_0.json"
            with open(log_file, "w") as f:
                json.dump(result, f, default=str, indent=2)
        
        return result

    def postprocess_sql_result(self, raw_result: Any) -> str:
        """Postprocess SQL execution result to prevent overly long responses.
        
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

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the SQL tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing sql_query

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        # Validate parameters
        logger.info(f"[SqlTool] Executing with parameters: {parameters}")
        is_valid, error_msg = self.validate_parameters(parameters)
        if not is_valid:
            error_response = json.dumps({"result": error_msg})
            logger.error(f"[SqlTool] Parameter validation failed: {error_msg}")
            return error_response, 0.0, {"error": error_msg}

        sql_query = parameters["sql_query"]

        try:
            # Get database information from instance data
            db_id, data_source = self.get_database_info_from_instance(instance_id)
            
            # Prepare database file
            db_file = self.prepare_database_file(db_id, data_source)
            
            # Set up logging directory for this execution
            execution_log_dir = None
            if self.enable_logging and self.log_dir:
                execution_log_dir = str(Path(self.log_dir) / f"sql_execution_{instance_id}")

            # Execute SQL query using Ray execution pool
            execution_result = await self.execution_pool.execute.remote(
                self.execute_sql_single,
                instance_id,
                sql_query,
                db_file,
                self.timeout,
                execution_log_dir
            )

            # Check if execution_result is None (indicating worker-level failure)
            if execution_result is None:
                execution_result = {
                    "success": False,
                    "error": "SQL execution failed at worker level",
                    "execution_result": None,
                    "execution_time": 0.0
                }

            # Update instance state
            self._instance_dict[instance_id]["execution_results"].append(execution_result)
            self._instance_dict[instance_id]["total_queries"] += 1
            
            # Calculate success metrics
            successful = execution_result.get("success", False)
            
            if successful:
                self._instance_dict[instance_id]["successful_queries"] += 1
            else:
                self._instance_dict[instance_id]["failed_queries"] += 1

            # Postprocess the execution result
            raw_result = execution_result.get("execution_result", None)
            
            if successful:
                # SQL执行成功，处理查询结果
                processed_result = self.postprocess_sql_result(raw_result)
            else:
                # SQL执行失败，返回错误信息
                if raw_result and isinstance(raw_result, str):
                    # 如果raw_result是错误字符串，直接使用
                    processed_result = raw_result
                else:
                    # 如果没有具体错误信息，使用通用错误信息
                    error_msg = execution_result.get("error", "Unknown SQL execution error")
                    processed_result = f"SQL execution failed: {error_msg}"

            response_data = {
                "result": processed_result,
            }
            # # Prepare response
            # response_data = {
            #     "result": execution_result,
            #     "summary": {
            #         "sql_query": sql_query,
            #         "database": db_id,
            #         "data_source": data_source,
            #         "successful": successful,
            #         "execution_time": execution_result.get("execution_time", 0.0)
            #     }
            # }

            response_text = json.dumps(response_data, indent=2)

            # Calculate metrics
            metrics = {
                "sql_query": sql_query,
                "database": db_id,
                "data_source": data_source,
                "successful": successful,
                "execution_time": execution_result.get("execution_time", 0.0),
                "error": execution_result.get("error") if not successful else None
            }

            # Reward calculation based on success
            reward_score = 1.0 if successful else 0.0

            logger.info(f"[SqlTool] Executed query, success: {successful}")

            return response_text, reward_score, metrics

        except Exception as e:
            error_response = json.dumps({"result": f"SQL execution failed: {str(e)}"})
            logger.error(f"[SqlTool] Execution failed: {e}")
            return error_response, 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward based on execution success rate."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance_data = self._instance_dict[instance_id]
        total = instance_data["total_queries"]
        successful = instance_data["successful_queries"]
        
        if total == 0:
            return 0.0
        
        return successful / total

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id] 