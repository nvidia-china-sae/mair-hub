#!/usr/bin/env python3
"""
Text2SQL 主评估脚本

该脚本负责：
1. 命令行接口
2. 配置参数解析
3. 并发评估管理
4. 进度监控
5. 结果保存和报告生成
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# 导入本地模块
try:
    # 尝试相对导入（作为包使用时）
    from .data_preprocess import DatabaseManager, PromptBuilder, DataProcessor
    from .conversation_manager import ConversationManager
    from .tool_client import SQLToolClient
    from .evaluator import Text2SQLEvaluator, EvaluationResult, EvaluationMetrics
    from .result_analyzer import ResultAnalyzer
except ImportError:
    # 回退到绝对导入（直接运行时）
    from data_preprocess import DatabaseManager, PromptBuilder, DataProcessor
    from conversation_manager import ConversationManager
    from tool_client import SQLToolClient
    from evaluator import Text2SQLEvaluator, EvaluationResult, EvaluationMetrics
    from result_analyzer import ResultAnalyzer

# 设置日志
logger = logging.getLogger(__name__)


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int):
        """
        初始化进度跟踪器
        
        Args:
            total: 总任务数
        """
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """
        更新进度
        
        Args:
            increment: 增量
        """
        self.completed += increment
        current_time = time.time()
        
        # 每秒最多更新一次
        if current_time - self.last_update >= 1.0:
            self.print_progress()
            self.last_update = current_time
    
    def print_progress(self):
        """打印进度信息"""
        if self.total <= 0:
            return
        
        percentage = (self.completed / self.total) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.completed > 0:
            eta = (elapsed_time / self.completed) * (self.total - self.completed)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        print(f"\rProgress: {self.completed}/{self.total} ({percentage:.1f}%) | "
              f"Time: {elapsed_time:.1f}s | {eta_str}", end="", flush=True)
    
    def finish(self):
        """完成进度跟踪"""
        self.print_progress()
        print()  # 换行


class Text2SQLEvaluationRunner:
    """Text2SQL评估运行器"""
    
    def __init__(self, args: argparse.Namespace):
        """
        初始化评估运行器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        
        # 初始化组件
        self.db_manager = DatabaseManager(args.db_root_path)
        self.prompt_builder = PromptBuilder(self.db_manager)
        self.data_processor = DataProcessor(self.db_manager, self.prompt_builder)
        self.sql_tool_client = SQLToolClient(
            db_root_path=args.db_root_path,
            timeout=args.sql_timeout,
            max_result_chars=args.max_result_chars,
            max_result_rows=args.max_result_rows
        )
        self.evaluator = Text2SQLEvaluator(self.sql_tool_client)
        self.result_analyzer = ResultAnalyzer()
        
        # 设置输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成运行ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized Text2SQL Evaluation Runner (Run ID: {self.run_id})")
    
    async def run_single_evaluation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个样本的评估，生成 n 条轨迹
        
        Args:
            sample: 样本数据
            
        Returns:
            评估结果（包含多条轨迹）
        """
        sample_id = sample.get('id', 'unknown')
        
        try:
            # 准备样本数据
            prepared_sample = self.data_processor.prepare_sample(sample)
            
            # 运行 n 次轨迹
            trajectories = []
            for trajectory_idx in range(self.args.n):
                try:
                    # 运行对话
                    async with ConversationManager(
                        server_url=self.args.server_url,
                        sql_tool_client=self.sql_tool_client,
                        max_turns=self.args.max_turns,
                        timeout=self.args.conversation_timeout,
                        model_name=self.args.model_name,
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                        stream=self.args.stream
                    ) as conversation_manager:
                        
                        conversation_result = await conversation_manager.run_conversation(
                            initial_messages=prepared_sample['messages'],
                            db_id=sample['db_id'],
                            data_source=sample['data_source']
                        )
                    
                    # 提取最终响应
                    if conversation_result.get('success', False):
                        final_response = conversation_result.get('final_response', '')
                        conversation_history = conversation_result.get('conversation_history', [])
                        turns = conversation_result.get('turns', 0)
                        tool_calls = conversation_result.get('tool_calls', 0)
                    else:
                        final_response = ''
                        conversation_history = conversation_result.get('conversation_history', [])
                        turns = conversation_result.get('turns', 0)
                        tool_calls = 0
                    
                    # 评估结果
                    eval_result, eval_metrics = self.evaluator.evaluate_single_sample(
                        prediction_text=final_response,
                        ground_truth_sql=sample['ground_truth_sql'],
                        db_id=sample['db_id'],
                        data_source=sample['data_source']
                    )
                    
                    # 构造轨迹结果
                    trajectory_result = {
                        'trajectory_id': trajectory_idx,
                        'success': conversation_result.get('success', False),
                        'final_response': final_response,
                        'conversation_history': conversation_history,
                        'turns': turns,
                        'tool_calls': tool_calls,
                        'evaluation_result': eval_result.value,
                        'evaluation_metrics': {
                            'execution_accuracy': eval_metrics.execution_accuracy,
                            'format_accuracy': eval_metrics.format_accuracy,
                            'sql_extracted': eval_metrics.sql_extracted,
                            'sql_valid': eval_metrics.sql_valid,
                            'prediction_success': eval_metrics.prediction_success,
                            'ground_truth_success': eval_metrics.ground_truth_success,
                            'execution_time': eval_metrics.execution_time,
                            'error_type': eval_metrics.error_type,
                            'error_message': eval_metrics.error_message
                        }
                    }
                    
                    trajectories.append(trajectory_result)
                    
                except Exception as e:
                    logger.error(f"Error in trajectory {trajectory_idx} for sample {sample_id}: {e}")
                    
                    # 添加错误轨迹结果
                    error_trajectory = {
                        'trajectory_id': trajectory_idx,
                        'success': False,
                        'error': str(e),
                        'evaluation_result': EvaluationResult.ERROR.value,
                        'evaluation_metrics': {
                            'execution_accuracy': False,
                            'format_accuracy': False,
                            'sql_extracted': False,
                            'sql_valid': False,
                            'prediction_success': False,
                            'ground_truth_success': False,
                            'execution_time': 0.0,
                            'error_type': 'trajectory_error',
                            'error_message': str(e)
                        }
                    }
                    trajectories.append(error_trajectory)
            
            # 计算样本级别的结果（只要有一条轨迹正确就算正确）
            sample_correct = any(
                traj['evaluation_result'] == EvaluationResult.CORRECT.value 
                for traj in trajectories
            )
            
            # 构造完整结果
            result = {
                'sample_id': sample_id,
                'trajectories': trajectories,
                'sample_correct': sample_correct,
                'n_trajectories': len(trajectories),
                'sample_data': sample
            }
            
            logger.debug(f"Completed evaluation for sample {sample_id}: {len(trajectories)} trajectories, sample_correct: {sample_correct}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample_id}: {e}")
            
            # 返回错误结果
            return {
                'sample_id': sample_id,
                'trajectories': [],
                'sample_correct': False,
                'n_trajectories': 0,
                'error': str(e),
                'sample_data': sample
            }
    
    async def run_evaluation_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        运行批量评估
        
        Args:
            samples: 样本列表
            
        Returns:
            评估结果列表
        """
        logger.info(f"Starting batch evaluation of {len(samples)} samples")
        
        # 创建进度跟踪器
        progress_tracker = ProgressTracker(len(samples))
        
        # 根据并发数量分批处理
        batch_size = self.args.concurrent_requests
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            # 并发执行批次
            batch_tasks = [self.run_single_evaluation(sample) for sample in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation error: {result}")
                    # 创建错误结果
                    error_result = {
                        'sample_id': 'unknown',
                        'trajectories': [],
                        'sample_correct': False,
                        'n_trajectories': 0,
                        'error': str(result)
                    }
                    results.append(error_result)
                else:
                    results.append(result)
                
                # 更新进度
                progress_tracker.update()
        
        progress_tracker.finish()
        logger.info(f"Completed batch evaluation")
        
        return results
    
    async def run_full_evaluation(self):
        """运行完整的评估流程"""
        logger.info(f"Starting full evaluation (Run ID: {self.run_id})")
        
        try:
            # 1. 加载数据
            logger.info("Loading dataset...")
            samples = self.data_processor.load_dataset(
                dataset_path=self.args.dataset_path,
                sample_size=self.args.sample_size
            )
            
            if not samples:
                logger.error("No samples loaded from dataset")
                return
            
            logger.info(f"Loaded {len(samples)} samples")
            
            # 2. 运行评估
            logger.info("Running evaluation...")
            results = await self.run_evaluation_batch(samples)
            
            # 3. 保存原始结果
            raw_results_path = self.output_dir / f"raw_results_{self.run_id}.json"
            with open(raw_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Raw results saved to {raw_results_path}")
            
            # 4. 分析结果
            logger.info("Analyzing results...")
            
            # 分析结果（传递新的数据结构）
            analysis = self.result_analyzer.analyze_results(results, samples)
            
            # 5. 保存分析结果
            analysis_path = self.output_dir / f"analysis_{self.run_id}.json"
            self.result_analyzer.save_detailed_results(str(analysis_path))
            
            # 6. 生成报告
            report_path = self.output_dir / f"report_{self.run_id}.txt"
            report = self.result_analyzer.generate_report(str(report_path))
            
            # 7. 打印摘要
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Run ID: {self.run_id}")
            print(f"Total Samples: {len(samples)}")
            print(f"Total Trajectories: {analysis['basic_statistics']['total_trajectories']}")
            print(f"Avg Trajectories per Sample: {analysis['basic_statistics']['avg_trajectories_per_sample']:.2f}")
            print(f"Sample-level Accuracy: {analysis['basic_statistics']['sample_accuracy']:.4f}")
            print(f"Trajectory-level Accuracy: {analysis['basic_statistics']['trajectory_accuracy']:.4f}")
            print(f"SQL Extraction Rate: {analysis['basic_statistics']['sql_extraction_rate']:.4f}")
            print(f"Results saved to: {self.output_dir}")
            print("="*60)
            
            logger.info("Evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="Text2SQL Evaluation Script")
    
    # 必需参数
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='spider_data/filter_test.json',
        help="Path to the evaluation dataset file"
    )
    
    parser.add_argument(
        "--db_root_path",
        type=str,
        default='spider_data',
        help="Root path to the database files"
    )
    
    parser.add_argument(
        "--server_url",
        type=str,
        default='http://localhost:30000',
        help="SGLang server URL (e.g., http://localhost:8001)"
    )
    
    # 可选参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="Maximum number of conversation turns (default: 6)"
    )
    
    parser.add_argument(
        "--concurrent_requests",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)"
    )
    
    parser.add_argument(
        "--conversation_timeout",
        type=int,
        default=300,
        help="Conversation timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--sql_timeout",
        type=int,
        default=30,
        help="SQL execution timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--max_result_chars",
        type=int,
        default=9000,
        help="Maximum characters in SQL result (default: 9000)"
    )
    
    parser.add_argument(
        "--max_result_rows",
        type=int,
        default=50,
        help="Maximum rows in SQL result (default: 50)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="huggingface_60",
        help="Model name for SGLang server (default: huggingface_60)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for model generation (default: 0.6)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=30000,
        help="Maximum tokens for model generation (default: 30000)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable streaming for model generation (default: False)"
    )
    
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of trajectories to generate per sample (default: 1)"
    )
    
    return parser


def setup_logging(log_level: str):
    """设置日志配置"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sql_eval.log')
        ]
    )


def validate_arguments(args: argparse.Namespace) -> bool:
    """验证命令行参数"""
    # 检查数据集文件
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset file not found: {args.dataset_path}")
        return False
    
    # 检查数据库根目录
    if not Path(args.db_root_path).exists():
        print(f"Error: Database root path not found: {args.db_root_path}")
        return False
    
    # 检查服务器URL格式
    if not args.server_url.startswith(('http://', 'https://')):
        print(f"Error: Invalid server URL format: {args.server_url}")
        return False
    
    # 检查数值参数
    if args.sample_size is not None and args.sample_size <= 0:
        print(f"Error: Sample size must be positive: {args.sample_size}")
        return False
    
    if args.max_turns <= 0:
        print(f"Error: Max turns must be positive: {args.max_turns}")
        return False
    
    if args.concurrent_requests <= 0:
        print(f"Error: Concurrent requests must be positive: {args.concurrent_requests}")
        return False
    
    # 检查模型参数
    if args.temperature < 0.0 or args.temperature > 2.0:
        print(f"Error: Temperature must be between 0.0 and 2.0: {args.temperature}")
        return False
    
    if args.max_tokens <= 0:
        print(f"Error: Max tokens must be positive: {args.max_tokens}")
        return False
    
    if args.n <= 0:
        print(f"Error: Number of trajectories must be positive: {args.n}")
        return False
    
    return True


async def main():
    """主函数"""
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 验证参数
    if not validate_arguments(args):
        sys.exit(1)
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 创建并运行评估器
    try:
        runner = Text2SQLEvaluationRunner(args)
        await runner.run_full_evaluation()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 