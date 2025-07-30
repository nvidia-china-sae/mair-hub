#!/usr/bin/env python3
"""
结果评估器模块

该模块负责：
1. SQL提取和验证
2. 执行结果比较
3. 多维度评估指标
4. 错误分析和分类
"""

import re
import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

try:
    from .tool_client import SQLToolClient, execute_sql_with_timeout
except ImportError:
    from tool_client import SQLToolClient, execute_sql_with_timeout

logger = logging.getLogger(__name__)


class EvaluationResult(Enum):
    """评估结果枚举"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    execution_accuracy: bool
    format_accuracy: bool
    sql_extracted: bool
    sql_valid: bool
    prediction_success: bool
    ground_truth_success: bool
    execution_time: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class SQLExtractor:
    """SQL提取器，从模型响应中提取SQL查询"""
    
    def __init__(self):
        """初始化SQL提取器"""
        self.sql_patterns = [
            # 匹配 <answer> 标签中的SQL
            r'<answer>\s*(.*?)\s*</answer>',
            # # 匹配SQL代码块
            # r'```sql\s*(.*?)\s*```',
            # # 匹配通用代码块
            # r'```\s*(.*?)\s*```',
            # # 匹配单行SQL语句
            # r'(?:^|\n)\s*(SELECT\s+.*?(?:;|$))',
            # r'(?:^|\n)\s*(INSERT\s+.*?(?:;|$))',
            # r'(?:^|\n)\s*(UPDATE\s+.*?(?:;|$))',
            # r'(?:^|\n)\s*(DELETE\s+.*?(?:;|$))',
            # r'(?:^|\n)\s*(CREATE\s+.*?(?:;|$))',
            # r'(?:^|\n)\s*(DROP\s+.*?(?:;|$))',
            # r'(?:^|\n)\s*(ALTER\s+.*?(?:;|$))',
        ]
    
    def extract_sql(self, response_text: str) -> Optional[str]:
        """
        从响应文本中提取SQL查询
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            提取的SQL查询，如果未找到则返回None
        """
        # print("response_text:", response_text)
        if not response_text:
            return None
        
        # 按优先级尝试各种模式
        for pattern in self.sql_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if matches:
                sql_query = matches[0].strip()
                
                # 清理SQL查询
                sql_query = self._clean_sql(sql_query)
                
                # 验证SQL是否有效
                if self._is_valid_sql(sql_query):
                    return sql_query
        
        return None
    
    def _clean_sql(self, sql_query: str) -> str:
        """
        清理SQL查询
        
        Args:
            sql_query: 原始SQL查询
            
        Returns:
            清理后的SQL查询
        """
        # 移除多余的空白字符
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        # 确保以分号结尾
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    def _is_valid_sql(self, sql_query: str) -> bool:
        """
        验证SQL查询是否有效
        
        Args:
            sql_query: SQL查询
            
        Returns:
            是否有效
        """
        if not sql_query:
            return False
        
        # 基本格式检查
        sql_upper = sql_query.upper().strip()
        
        # 检查是否包含SQL关键字
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        has_keyword = any(sql_upper.startswith(keyword) for keyword in sql_keywords)
        
        if not has_keyword:
            return False
        
        # 检查括号匹配
        if sql_query.count('(') != sql_query.count(')'):
            return False
        
        # 检查引号匹配
        if sql_query.count("'") % 2 != 0:
            return False
        
        if sql_query.count('"') % 2 != 0:
            return False
        
        return True


class Text2SQLEvaluator:
    """Text2SQL评估器，实现SQL结果的比较和评估"""
    
    def __init__(self, sql_tool_client: SQLToolClient):
        """
        初始化评估器
        
        Args:
            sql_tool_client: SQL工具客户端
        """
        self.sql_tool_client = sql_tool_client
        self.sql_extractor = SQLExtractor()
    
    def execute_sql_for_evaluation(self, sql_query: str, db_id: str, data_source: str) -> Tuple[Any, bool, str]:
        """
        为评估目的执行SQL查询
        
        Args:
            sql_query: SQL查询
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            (执行结果, 是否成功, 错误信息)
        """
        try:
            # 获取数据库文件路径
            db_file = self.sql_tool_client.get_database_file_path(db_id, data_source)
            
            # 执行SQL查询，使用新的execute_sql_with_timeout函数
            execution_result = execute_sql_with_timeout(0, db_file, sql_query, self.sql_tool_client.timeout, "")
            
            # 解析执行结果 - 格式: (data_idx, db_file, sql, execution_res, success, output_str)
            data_idx, db_file_result, sql_result, execution_res, success, output_str = execution_result
            
            # 转换成功标志为布尔值
            success = bool(success)
            
            # 处理错误信息
            error = ""
            if not success and isinstance(execution_res, str):
                error = execution_res
            
            return execution_res, success, error
            
        except Exception as e:
            logger.error(f"Error executing SQL for evaluation: {e}")
            return None, False, str(e)
    
    def compare_sql_results(self, result1: Any, result2: Any) -> bool:
        """
        比较两个SQL执行结果是否相同
        
        Args:
            result1: 第一个结果
            result2: 第二个结果
            
        Returns:
            结果是否相同
        """
        try:
            # 处理None结果
            if result1 is None and result2 is None:
                return True
            if result1 is None or result2 is None:
                return False
            
            # 处理frozenset结果
            if isinstance(result1, frozenset) and isinstance(result2, frozenset):
                return result1 == result2
            
            # 转换为frozenset进行比较
            if isinstance(result1, (list, tuple, set)):
                result1 = frozenset(result1)
            if isinstance(result2, (list, tuple, set)):
                result2 = frozenset(result2)
            
            # 如果都是frozenset，直接比较
            if isinstance(result1, frozenset) and isinstance(result2, frozenset):
                return result1 == result2
            
            # 其他类型的比较
            return result1 == result2
            
        except Exception as e:
            logger.error(f"Error comparing SQL results: {e}")
            return False
    
    def evaluate_single_sample(self, prediction_text: str, ground_truth_sql: str, 
                             db_id: str, data_source: str) -> Tuple[EvaluationResult, EvaluationMetrics]:
        """
        评估单个样本
        
        Args:
            prediction_text: 模型预测文本
            ground_truth_sql: 真实SQL查询
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            (评估结果, 评估指标)
        """
        import time
        
        start_time = time.time()
        
        # 初始化指标
        metrics = EvaluationMetrics(
            execution_accuracy=False,
            format_accuracy=False,
            sql_extracted=False,
            sql_valid=False,
            prediction_success=False,
            ground_truth_success=False,
            execution_time=0.0
        )
        
        try:
            # 1. 提取SQL
            predicted_sql = self.sql_extractor.extract_sql(prediction_text)
            
            if predicted_sql is None:
                metrics.execution_time = time.time() - start_time
                metrics.error_type = "parse_error"
                metrics.error_message = "Failed to extract SQL from prediction"
                return EvaluationResult.PARSE_ERROR, metrics
            
            metrics.sql_extracted = True
            metrics.sql_valid = self.sql_extractor._is_valid_sql(predicted_sql)
            
            # 2. 执行预测SQL
            pred_result, pred_success, pred_error = self.execute_sql_for_evaluation(
                predicted_sql, db_id, data_source
            )
            metrics.prediction_success = pred_success
            
            # 3. 执行真实SQL
            gt_result, gt_success, gt_error = self.execute_sql_for_evaluation(
                ground_truth_sql, db_id, data_source
            )
            metrics.ground_truth_success = gt_success
            
            # 4. 比较结果
            if pred_success and gt_success:
                # 两个查询都成功执行，比较结果
                results_match = self.compare_sql_results(pred_result, gt_result)
                metrics.execution_accuracy = results_match
                
                if results_match:
                    metrics.execution_time = time.time() - start_time
                    return EvaluationResult.CORRECT, metrics
                else:
                    metrics.execution_time = time.time() - start_time
                    metrics.error_type = "incorrect_result"
                    metrics.error_message = "Query results do not match"
                    return EvaluationResult.INCORRECT, metrics
            
            elif not pred_success and not gt_success:
                # 两个查询都失败，检查是否是相同的错误
                metrics.execution_accuracy = True  # 都失败也算是一种匹配
                metrics.execution_time = time.time() - start_time
                return EvaluationResult.CORRECT, metrics
            
            else:
                # 一个成功一个失败
                metrics.execution_accuracy = False
                metrics.execution_time = time.time() - start_time
                metrics.error_type = "execution_mismatch"
                
                if not pred_success:
                    metrics.error_message = f"Prediction failed: {pred_error}"
                else:
                    metrics.error_message = f"Ground truth failed: {gt_error}"
                
                return EvaluationResult.INCORRECT, metrics
        
        except Exception as e:
            metrics.execution_time = time.time() - start_time
            metrics.error_type = "evaluation_error"
            metrics.error_message = str(e)
            logger.error(f"Error in evaluation: {e}")
            return EvaluationResult.ERROR, metrics
    
    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> List[Tuple[EvaluationResult, EvaluationMetrics]]:
        """
        批量评估样本
        
        Args:
            samples: 样本列表，每个样本包含预测结果和真实答案
            
        Returns:
            评估结果列表
        """
        results = []
        
        for i, sample in enumerate(samples):
            try:
                logger.info(f"Evaluating sample {i+1}/{len(samples)}")
                
                prediction_text = sample.get('prediction', '')
                ground_truth_sql = sample.get('ground_truth_sql', '')
                db_id = sample.get('db_id', '')
                data_source = sample.get('data_source', 'synsql')
                
                if not prediction_text or not ground_truth_sql or not db_id:
                    logger.warning(f"Sample {i+1} missing required fields")
                    metrics = EvaluationMetrics(
                        execution_accuracy=False,
                        format_accuracy=False,
                        sql_extracted=False,
                        sql_valid=False,
                        prediction_success=False,
                        ground_truth_success=False,
                        execution_time=0.0,
                        error_type="missing_fields",
                        error_message="Sample missing required fields"
                    )
                    results.append((EvaluationResult.ERROR, metrics))
                    continue
                
                # 评估单个样本
                eval_result, eval_metrics = self.evaluate_single_sample(
                    prediction_text, ground_truth_sql, db_id, data_source
                )
                
                results.append((eval_result, eval_metrics))
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i+1}: {e}")
                metrics = EvaluationMetrics(
                    execution_accuracy=False,
                    format_accuracy=False,
                    sql_extracted=False,
                    sql_valid=False,
                    prediction_success=False,
                    ground_truth_success=False,
                    execution_time=0.0,
                    error_type="evaluation_error",
                    error_message=str(e)
                )
                results.append((EvaluationResult.ERROR, metrics))
        
        return results
    
    def compute_aggregate_metrics(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]]) -> Dict[str, Any]:
        """
        计算聚合指标
        
        Args:
            results: 评估结果列表
            
        Returns:
            聚合指标字典
        """
        if not results:
            return {}
        
        total_samples = len(results)
        
        # 统计各种结果
        correct_count = sum(1 for result, _ in results if result == EvaluationResult.CORRECT)
        incorrect_count = sum(1 for result, _ in results if result == EvaluationResult.INCORRECT)
        error_count = sum(1 for result, _ in results if result == EvaluationResult.ERROR)
        timeout_count = sum(1 for result, _ in results if result == EvaluationResult.TIMEOUT)
        parse_error_count = sum(1 for result, _ in results if result == EvaluationResult.PARSE_ERROR)
        
        # 统计指标
        metrics_list = [metrics for _, metrics in results]
        
        sql_extracted_count = sum(1 for m in metrics_list if m.sql_extracted)
        sql_valid_count = sum(1 for m in metrics_list if m.sql_valid)
        prediction_success_count = sum(1 for m in metrics_list if m.prediction_success)
        ground_truth_success_count = sum(1 for m in metrics_list if m.ground_truth_success)
        execution_accuracy_count = sum(1 for m in metrics_list if m.execution_accuracy)
        
        # 计算执行时间统计
        execution_times = [m.execution_time for m in metrics_list if m.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        # 错误类型统计
        error_types = {}
        for metrics in metrics_list:
            if metrics.error_type:
                error_types[metrics.error_type] = error_types.get(metrics.error_type, 0) + 1
        
        return {
            # 基本统计
            "total_samples": total_samples,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "error_count": error_count,
            "timeout_count": timeout_count,
            "parse_error_count": parse_error_count,
            
            # 准确率指标
            "execution_accuracy": correct_count / total_samples if total_samples > 0 else 0.0,
            "sql_extraction_rate": sql_extracted_count / total_samples if total_samples > 0 else 0.0,
            "sql_validity_rate": sql_valid_count / total_samples if total_samples > 0 else 0.0,
            "prediction_success_rate": prediction_success_count / total_samples if total_samples > 0 else 0.0,
            "ground_truth_success_rate": ground_truth_success_count / total_samples if total_samples > 0 else 0.0,
            
            # 性能指标
            "avg_execution_time": avg_execution_time,
            "total_execution_time": sum(execution_times),
            
            # 错误分析
            "error_types": error_types,
            
            # 详细分布
            "result_distribution": {
                "correct": correct_count,
                "incorrect": incorrect_count,
                "error": error_count,
                "timeout": timeout_count,
                "parse_error": parse_error_count
            }
        }


def test_evaluator():
    """测试评估器功能"""
    import tempfile
    import os
    
    # 创建临时数据库用于测试
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
        temp_db = f.name
    
    try:
        # 创建测试数据库
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # 创建测试表
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER
            )
        """)
        
        # 插入测试数据
        cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 25)")
        cursor.execute("INSERT INTO users (name, age) VALUES ('Bob', 30)")
        cursor.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)")
        
        conn.commit()
        conn.close()
        
        # 创建临时目录结构
        temp_dir = tempfile.mkdtemp()
        test_db_dir = Path(temp_dir) / "synsql" / "databases" / "test_db"
        test_db_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制数据库文件
        import shutil
        shutil.copy2(temp_db, test_db_dir / "test_db.sqlite")
        
        # 测试评估器
        from .tool_client import SQLToolClient
        
        sql_client = SQLToolClient(temp_dir)
        evaluator = Text2SQLEvaluator(sql_client)
        
        # 测试SQL提取
        response_text = """
        <answer>
        SELECT COUNT(*) FROM users;
        </answer>
        """
        
        extracted_sql = evaluator.sql_extractor.extract_sql(response_text)
        print(f"Extracted SQL: {extracted_sql}")
        
        # 测试评估
        result, metrics = evaluator.evaluate_single_sample(
            response_text,
            "SELECT COUNT(*) FROM users;",
            "test_db",
            "synsql"
        )
        
        print(f"Evaluation result: {result}")
        print(f"Metrics: {metrics}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_db):
            os.unlink(temp_db)
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_evaluator() 