#!/usr/bin/env python3
"""
结果分析器模块

该模块负责：
1. 统计分析（准确率、成功率等）
2. 错误类型分析
3. 性能分析（延迟、吞吐量等）
4. 结果报告生成
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import statistics

try:
    from .evaluator import EvaluationResult, EvaluationMetrics
except ImportError:
    from evaluator import EvaluationResult, EvaluationMetrics

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """结果分析器，负责分析评估结果并生成报告"""
    
    def __init__(self):
        """初始化结果分析器"""
        self.analysis_results: Dict[str, Any] = {}
    
    def analyze_results(self, evaluation_results: List[Dict[str, Any]], 
                       sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析评估结果（支持多轨迹）
        
        Args:
            evaluation_results: 评估结果列表（新格式，包含多轨迹）
            sample_data: 样本数据列表
            
        Returns:
            分析结果字典
        """
        if not evaluation_results:
            return {"error": "No evaluation results to analyze"}
        
        # 基本统计分析（样本级别和轨迹级别）
        basic_stats = self._analyze_basic_statistics_multi_trajectory(evaluation_results)
        
        # 错误类型分析
        error_analysis = self._analyze_error_types_multi_trajectory(evaluation_results)
        
        # 性能分析
        performance_analysis = self._analyze_performance_multi_trajectory(evaluation_results)
        
        # 难度分析（如果有难度信息）
        difficulty_analysis = self._analyze_by_difficulty_multi_trajectory(evaluation_results, sample_data)
        
        # 数据源分析
        data_source_analysis = self._analyze_by_data_source_multi_trajectory(evaluation_results, sample_data)
        
        # 时间分析
        time_analysis = self._analyze_time_distribution_multi_trajectory(evaluation_results)
        
        # 综合分析
        comprehensive_analysis = self._comprehensive_analysis_multi_trajectory(evaluation_results, sample_data)
        
        # 构建完整分析结果
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(evaluation_results),
            "basic_statistics": basic_stats,
            "error_analysis": error_analysis,
            "performance_analysis": performance_analysis,
            "difficulty_analysis": difficulty_analysis,
            "data_source_analysis": data_source_analysis,
            "time_analysis": time_analysis,
            "comprehensive_analysis": comprehensive_analysis
        }
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def _analyze_basic_statistics_multi_trajectory(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析基本统计信息（多轨迹版本）"""
        total_samples = len(results)
        
        # 样本级别统计
        sample_correct_count = sum(1 for r in results if r.get('sample_correct', False))
        sample_accuracy = sample_correct_count / total_samples if total_samples > 0 else 0.0
        
        # 轨迹级别统计
        all_trajectories = []
        for result in results:
            trajectories = result.get('trajectories', [])
            all_trajectories.extend(trajectories)
        
        total_trajectories = len(all_trajectories)
        
        if total_trajectories > 0:
            # 统计各种结果类型
            result_counts = {}
            for result_type in EvaluationResult:
                result_counts[result_type.value] = sum(
                    1 for traj in all_trajectories 
                    if traj.get('evaluation_result') == result_type.value
                )
            
            # 计算轨迹级别准确率
            trajectory_correct_count = result_counts.get('correct', 0)
            trajectory_accuracy = trajectory_correct_count / total_trajectories
            
            # 统计指标
            sql_extracted_count = sum(
                1 for traj in all_trajectories 
                if traj.get('evaluation_metrics', {}).get('sql_extracted', False)
            )
            sql_valid_count = sum(
                1 for traj in all_trajectories 
                if traj.get('evaluation_metrics', {}).get('sql_valid', False)
            )
            prediction_success_count = sum(
                1 for traj in all_trajectories 
                if traj.get('evaluation_metrics', {}).get('prediction_success', False)
            )
            ground_truth_success_count = sum(
                1 for traj in all_trajectories 
                if traj.get('evaluation_metrics', {}).get('ground_truth_success', False)
            )
            
            # 计算平均轨迹数
            avg_trajectories_per_sample = total_trajectories / total_samples
            
        else:
            result_counts = {}
            trajectory_accuracy = 0.0
            sql_extracted_count = 0
            sql_valid_count = 0
            prediction_success_count = 0
            ground_truth_success_count = 0
            avg_trajectories_per_sample = 0.0
        
        return {
            "total_samples": total_samples,
            "total_trajectories": total_trajectories,
            "avg_trajectories_per_sample": avg_trajectories_per_sample,
            
            # 样本级别统计
            "sample_accuracy": sample_accuracy,
            "sample_correct_count": sample_correct_count,
            
            # 轨迹级别统计
            "trajectory_accuracy": trajectory_accuracy,
            "trajectory_correct_count": trajectory_correct_count,
            "trajectory_result_distribution": result_counts,
            
            # 其他指标
            "sql_extraction_rate": sql_extracted_count / total_trajectories if total_trajectories > 0 else 0.0,
            "sql_validity_rate": sql_valid_count / total_trajectories if total_trajectories > 0 else 0.0,
            "prediction_success_rate": prediction_success_count / total_trajectories if total_trajectories > 0 else 0.0,
            "ground_truth_success_rate": ground_truth_success_count / total_trajectories if total_trajectories > 0 else 0.0
        }
    
    def _analyze_error_types_multi_trajectory(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析错误类型（多轨迹版本）"""
        error_types = {}
        error_messages = []
        
        for result in results:
            trajectories = result.get('trajectories', [])
            for traj in trajectories:
                metrics = traj.get('evaluation_metrics', {})
                error_type = metrics.get('error_type')
                error_message = metrics.get('error_message')
                
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    if error_message:
                        error_messages.append({
                            "type": error_type,
                            "message": error_message
                        })
        
        # 分析最常见的错误
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "error_type_distribution": error_types,
            "most_common_errors": most_common_errors,
            "total_errors": len(error_messages),
            "error_examples": error_messages[:20]  # 取前20个错误示例
        }
    
    def _analyze_performance_multi_trajectory(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能指标（多轨迹版本）"""
        execution_times = []
        
        for result in results:
            trajectories = result.get('trajectories', [])
            for traj in trajectories:
                metrics = traj.get('evaluation_metrics', {})
                exec_time = metrics.get('execution_time', 0)
                if exec_time > 0:
                    execution_times.append(exec_time)
        
        if not execution_times:
            return {"error": "No execution time data available"}
        
        # 计算统计指标
        avg_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        return {
            "avg_execution_time": avg_time,
            "median_execution_time": median_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "total_execution_time": sum(execution_times),
            "samples_with_time_data": len(execution_times)
        }
    
    def _analyze_by_difficulty_multi_trajectory(self, results: List[Dict[str, Any]], 
                                               sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按难度分析结果（多轨迹版本）"""
        return {"note": "Difficulty analysis not implemented for multi-trajectory"}
    
    def _analyze_by_data_source_multi_trajectory(self, results: List[Dict[str, Any]], 
                                                sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按数据源分析结果（多轨迹版本）"""
        return {"note": "Data source analysis not implemented for multi-trajectory"}
    
    def _analyze_time_distribution_multi_trajectory(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间分布（多轨迹版本）"""
        return {"note": "Time distribution analysis not implemented for multi-trajectory"}
    
    def _comprehensive_analysis_multi_trajectory(self, results: List[Dict[str, Any]], 
                                                sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """综合分析（多轨迹版本）"""
        return {"note": "Comprehensive analysis not implemented for multi-trajectory"}
    
    def _analyze_basic_statistics(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]]) -> Dict[str, Any]:
        """分析基本统计信息"""
        total_samples = len(results)
        
        # 统计各种结果类型
        result_counts = {}
        for result_type in EvaluationResult:
            result_counts[result_type.value] = sum(1 for r, _ in results if r == result_type)
        
        # 计算准确率
        correct_count = result_counts.get('correct', 0)
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        
        # 统计指标
        metrics_list = [metrics for _, metrics in results]
        
        sql_extracted_count = sum(1 for m in metrics_list if m.sql_extracted)
        sql_valid_count = sum(1 for m in metrics_list if m.sql_valid)
        prediction_success_count = sum(1 for m in metrics_list if m.prediction_success)
        ground_truth_success_count = sum(1 for m in metrics_list if m.ground_truth_success)
        
        return {
            "total_samples": total_samples,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "result_distribution": result_counts,
            "sql_extraction_rate": sql_extracted_count / total_samples if total_samples > 0 else 0.0,
            "sql_validity_rate": sql_valid_count / total_samples if total_samples > 0 else 0.0,
            "prediction_success_rate": prediction_success_count / total_samples if total_samples > 0 else 0.0,
            "ground_truth_success_rate": ground_truth_success_count / total_samples if total_samples > 0 else 0.0
        }
    
    def _analyze_error_types(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]]) -> Dict[str, Any]:
        """分析错误类型"""
        error_types = {}
        error_messages = []
        
        for result, metrics in results:
            if metrics.error_type:
                error_types[metrics.error_type] = error_types.get(metrics.error_type, 0) + 1
                
                if metrics.error_message:
                    error_messages.append({
                        "type": metrics.error_type,
                        "message": metrics.error_message
                    })
        
        # 分析最常见的错误
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "error_type_distribution": error_types,
            "most_common_errors": most_common_errors,
            "total_errors": len(error_messages),
            "error_examples": error_messages[:20]  # 取前20个错误示例
        }
    
    def _analyze_performance(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]]) -> Dict[str, Any]:
        """分析性能指标"""
        execution_times = []
        
        for result, metrics in results:
            if metrics.execution_time > 0:
                execution_times.append(metrics.execution_time)
        
        if not execution_times:
            return {"error": "No execution time data available"}
        
        # 计算统计指标
        avg_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # 计算百分位数
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            percentiles[f"p{p}"] = statistics.quantiles(execution_times, n=100)[p-1] if len(execution_times) > 1 else execution_times[0]
        
        # 时间分布
        time_ranges = {
            "fast": 0,      # < 1秒
            "medium": 0,    # 1-5秒
            "slow": 0,      # 5-30秒
            "very_slow": 0  # > 30秒
        }
        
        for time in execution_times:
            if time < 1.0:
                time_ranges["fast"] += 1
            elif time < 5.0:
                time_ranges["medium"] += 1
            elif time < 30.0:
                time_ranges["slow"] += 1
            else:
                time_ranges["very_slow"] += 1
        
        return {
            "avg_execution_time": avg_time,
            "median_execution_time": median_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "total_execution_time": sum(execution_times),
            "percentiles": percentiles,
            "time_distribution": time_ranges,
            "samples_with_time_data": len(execution_times)
        }
    
    def _analyze_by_difficulty(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]], 
                             sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按难度分析结果"""
        if len(results) != len(sample_data):
            return {"error": "Results and sample data length mismatch"}
        
        difficulty_stats = {}
        
        for (result, metrics), sample in zip(results, sample_data):
            difficulty = sample.get('difficulty', 'unknown')
            
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "error": 0,
                    "execution_times": []
                }
            
            difficulty_stats[difficulty]["total"] += 1
            
            if result == EvaluationResult.CORRECT:
                difficulty_stats[difficulty]["correct"] += 1
            elif result == EvaluationResult.INCORRECT:
                difficulty_stats[difficulty]["incorrect"] += 1
            else:
                difficulty_stats[difficulty]["error"] += 1
            
            if metrics.execution_time > 0:
                difficulty_stats[difficulty]["execution_times"].append(metrics.execution_time)
        
        # 计算每个难度的准确率
        for difficulty, stats in difficulty_stats.items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
                
                if stats["execution_times"]:
                    stats["avg_execution_time"] = statistics.mean(stats["execution_times"])
                else:
                    stats["avg_execution_time"] = 0.0
                
                # 移除执行时间列表以减少输出大小
                del stats["execution_times"]
        
        return difficulty_stats
    
    def _analyze_by_data_source(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]], 
                               sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """按数据源分析结果"""
        if len(results) != len(sample_data):
            return {"error": "Results and sample data length mismatch"}
        
        data_source_stats = {}
        
        for (result, metrics), sample in zip(results, sample_data):
            data_source = sample.get('data_source', 'unknown')
            
            if data_source not in data_source_stats:
                data_source_stats[data_source] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "error": 0,
                    "execution_times": []
                }
            
            data_source_stats[data_source]["total"] += 1
            
            if result == EvaluationResult.CORRECT:
                data_source_stats[data_source]["correct"] += 1
            elif result == EvaluationResult.INCORRECT:
                data_source_stats[data_source]["incorrect"] += 1
            else:
                data_source_stats[data_source]["error"] += 1
            
            if metrics.execution_time > 0:
                data_source_stats[data_source]["execution_times"].append(metrics.execution_time)
        
        # 计算每个数据源的准确率
        for data_source, stats in data_source_stats.items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
                
                if stats["execution_times"]:
                    stats["avg_execution_time"] = statistics.mean(stats["execution_times"])
                else:
                    stats["avg_execution_time"] = 0.0
                
                # 移除执行时间列表以减少输出大小
                del stats["execution_times"]
        
        return data_source_stats
    
    def _analyze_time_distribution(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]]) -> Dict[str, Any]:
        """分析时间分布"""
        execution_times = [metrics.execution_time for _, metrics in results if metrics.execution_time > 0]
        
        if not execution_times:
            return {"error": "No execution time data available"}
        
        # 创建时间桶
        max_time = max(execution_times)
        bucket_size = max_time / 10  # 分成10个桶
        
        time_buckets = {}
        for i in range(10):
            bucket_start = i * bucket_size
            bucket_end = (i + 1) * bucket_size
            bucket_key = f"{bucket_start:.2f}-{bucket_end:.2f}s"
            time_buckets[bucket_key] = 0
        
        # 分配时间到桶中
        for time in execution_times:
            bucket_index = min(int(time / bucket_size), 9)
            bucket_start = bucket_index * bucket_size
            bucket_end = (bucket_index + 1) * bucket_size
            bucket_key = f"{bucket_start:.2f}-{bucket_end:.2f}s"
            time_buckets[bucket_key] += 1
        
        return {
            "time_buckets": time_buckets,
            "bucket_size": bucket_size,
            "total_samples": len(execution_times)
        }
    
    def _comprehensive_analysis(self, results: List[Tuple[EvaluationResult, EvaluationMetrics]], 
                              sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """综合分析"""
        total_samples = len(results)
        
        # 成功率分析
        success_metrics = {
            "sql_extraction": 0,
            "sql_validity": 0,
            "prediction_execution": 0,
            "ground_truth_execution": 0,
            "result_match": 0
        }
        
        for result, metrics in results:
            if metrics.sql_extracted:
                success_metrics["sql_extraction"] += 1
            if metrics.sql_valid:
                success_metrics["sql_validity"] += 1
            if metrics.prediction_success:
                success_metrics["prediction_execution"] += 1
            if metrics.ground_truth_success:
                success_metrics["ground_truth_execution"] += 1
            if metrics.execution_accuracy:
                success_metrics["result_match"] += 1
        
        # 转换为百分比
        success_rates = {}
        for metric, count in success_metrics.items():
            success_rates[metric] = count / total_samples if total_samples > 0 else 0.0
        
        # 识别主要失败原因
        failure_reasons = []
        
        if success_rates["sql_extraction"] < 0.9:
            failure_reasons.append("SQL extraction failure")
        if success_rates["sql_validity"] < 0.9:
            failure_reasons.append("SQL validity issues")
        if success_rates["prediction_execution"] < 0.9:
            failure_reasons.append("Prediction execution failure")
        if success_rates["result_match"] < 0.8:
            failure_reasons.append("Result mismatch")
        
        # 性能评估
        execution_times = [metrics.execution_time for _, metrics in results if metrics.execution_time > 0]
        performance_grade = "A"
        
        if execution_times:
            avg_time = statistics.mean(execution_times)
            if avg_time > 10.0:
                performance_grade = "D"
            elif avg_time > 5.0:
                performance_grade = "C"
            elif avg_time > 2.0:
                performance_grade = "B"
        
        return {
            "success_rates": success_rates,
            "main_failure_reasons": failure_reasons,
            "performance_grade": performance_grade,
            "recommendations": self._generate_recommendations(success_rates, failure_reasons)
        }
    
    def _generate_recommendations(self, success_rates: Dict[str, float], 
                                failure_reasons: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if success_rates["sql_extraction"] < 0.9:
            recommendations.append("Improve SQL extraction logic or modify model output format")
        
        if success_rates["sql_validity"] < 0.9:
            recommendations.append("Enhance SQL validation or provide better training examples")
        
        if success_rates["prediction_execution"] < 0.9:
            recommendations.append("Review database schema access and SQL query complexity")
        
        if success_rates["result_match"] < 0.8:
            recommendations.append("Investigate semantic equivalence of SQL queries")
        
        if "Result mismatch" in failure_reasons:
            recommendations.append("Consider implementing approximate result matching")
        
        return recommendations
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        生成分析报告
        
        Args:
            output_path: 输出文件路径（可选）
            
        Returns:
            报告文本
        """
        if not self.analysis_results:
            return "No analysis results available. Please run analyze_results first."
        
        report_lines = []
        
        # 标题
        report_lines.append("=" * 60)
        report_lines.append("Text2SQL Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {self.analysis_results['timestamp']}")
        report_lines.append(f"Total samples: {self.analysis_results['total_samples']}")
        report_lines.append("")
        
        # 基本统计
        basic_stats = self.analysis_results["basic_statistics"]
        report_lines.append("Basic Statistics:")
        report_lines.append("-" * 20)
        
        # 检查是否是多轨迹分析
        if 'sample_accuracy' in basic_stats:
            # 多轨迹分析
            report_lines.append(f"Total Trajectories: {basic_stats['total_trajectories']}")
            report_lines.append(f"Avg Trajectories per Sample: {basic_stats['avg_trajectories_per_sample']:.2f}")
            report_lines.append(f"Sample-level Accuracy: {basic_stats['sample_accuracy']:.4f}")
            report_lines.append(f"Trajectory-level Accuracy: {basic_stats['trajectory_accuracy']:.4f}")
            report_lines.append(f"Sample Correct: {basic_stats['sample_correct_count']}")
            report_lines.append(f"Trajectory Correct: {basic_stats['trajectory_correct_count']}")
        else:
            # 单轨迹分析
            report_lines.append(f"Overall Accuracy: {basic_stats['accuracy']:.4f}")
            report_lines.append(f"Correct: {basic_stats['correct_count']}")
        
        report_lines.append(f"SQL Extraction Rate: {basic_stats['sql_extraction_rate']:.4f}")
        report_lines.append(f"SQL Validity Rate: {basic_stats['sql_validity_rate']:.4f}")
        report_lines.append(f"Prediction Success Rate: {basic_stats['prediction_success_rate']:.4f}")
        report_lines.append("")
        
        # 性能分析
        perf_analysis = self.analysis_results["performance_analysis"]
        if "error" not in perf_analysis:
            report_lines.append("Performance Analysis:")
            report_lines.append("-" * 20)
            report_lines.append(f"Average Execution Time: {perf_analysis['avg_execution_time']:.4f}s")
            report_lines.append(f"Median Execution Time: {perf_analysis['median_execution_time']:.4f}s")
            if 'percentiles' in perf_analysis:
                report_lines.append(f"95th Percentile: {perf_analysis['percentiles']['p95']:.4f}s")
            report_lines.append("")
        
        # 错误分析
        error_analysis = self.analysis_results["error_analysis"]
        if error_analysis["most_common_errors"]:
            report_lines.append("Most Common Errors:")
            report_lines.append("-" * 20)
            for error_type, count in error_analysis["most_common_errors"][:5]:
                report_lines.append(f"{error_type}: {count}")
            report_lines.append("")
        
        # 综合分析
        comprehensive = self.analysis_results["comprehensive_analysis"]
        report_lines.append("Comprehensive Analysis:")
        report_lines.append("-" * 20)
        
        if 'performance_grade' in comprehensive:
            report_lines.append(f"Performance Grade: {comprehensive['performance_grade']}")
        
        if comprehensive.get("main_failure_reasons"):
            report_lines.append("Main Failure Reasons:")
            for reason in comprehensive["main_failure_reasons"]:
                report_lines.append(f"  - {reason}")
        
        if comprehensive.get("recommendations"):
            report_lines.append("Recommendations:")
            for rec in comprehensive["recommendations"]:
                report_lines.append(f"  - {rec}")
        
        if comprehensive.get("note"):
            report_lines.append(f"Note: {comprehensive['note']}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def save_detailed_results(self, output_path: str):
        """
        保存详细分析结果到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        if not self.analysis_results:
            logger.warning("No analysis results to save")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {output_path}")


def test_result_analyzer():
    """测试结果分析器功能"""
    from .evaluator import EvaluationResult, EvaluationMetrics
    
    # 创建测试数据
    test_results = [
        (EvaluationResult.CORRECT, EvaluationMetrics(
            execution_accuracy=True,
            format_accuracy=True,
            sql_extracted=True,
            sql_valid=True,
            prediction_success=True,
            ground_truth_success=True,
            execution_time=1.2
        )),
        (EvaluationResult.INCORRECT, EvaluationMetrics(
            execution_accuracy=False,
            format_accuracy=True,
            sql_extracted=True,
            sql_valid=True,
            prediction_success=True,
            ground_truth_success=True,
            execution_time=2.5,
            error_type="incorrect_result",
            error_message="Results do not match"
        )),
        (EvaluationResult.PARSE_ERROR, EvaluationMetrics(
            execution_accuracy=False,
            format_accuracy=False,
            sql_extracted=False,
            sql_valid=False,
            prediction_success=False,
            ground_truth_success=True,
            execution_time=0.5,
            error_type="parse_error",
            error_message="Failed to extract SQL"
        ))
    ]
    
    test_samples = [
        {"difficulty": "easy", "data_source": "synsql"},
        {"difficulty": "medium", "data_source": "spider"},
        {"difficulty": "hard", "data_source": "bird"}
    ]
    
    # 测试分析器
    analyzer = ResultAnalyzer()
    analysis = analyzer.analyze_results(test_results, test_samples)
    
    print("Analysis Results:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # 生成报告
    report = analyzer.generate_report()
    print("\nGenerated Report:")
    print(report)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_result_analyzer() 