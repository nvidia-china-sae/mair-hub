#!/usr/bin/env python3
"""
Text2SQL 评估包

该包提供了完整的Text2SQL评估功能，包括：
1. 数据预处理和Prompt构造
2. 多轮对话管理
3. SQL工具调用
4. 结果评估和分析
5. 报告生成

主要模块：
- data_preprocess: 数据预处理和Prompt构造
- conversation_manager: 多轮对话管理
- tool_client: SQL工具调用客户端
- evaluator: 结果评估器
- result_analyzer: 结果分析器
- main_eval: 主评估脚本
"""

__version__ = "1.0.0"
__author__ = "Text2SQL Evaluation Team"
__email__ = "team@example.com"

# 导入主要类和函数
from .data_preprocess import (
    DatabaseManager,
    PromptBuilder,
    DataProcessor
)

from .conversation_manager import (
    ConversationManager,
    ConversationState
)

from .tool_client import (
    SQLToolClient
)

from .evaluator import (
    Text2SQLEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    SQLExtractor
)

from .result_analyzer import (
    ResultAnalyzer
)

# 定义公共API
__all__ = [
    # 数据处理
    'DatabaseManager',
    'PromptBuilder', 
    'DataProcessor',
    
    # 对话管理
    'ConversationManager',
    'ConversationState',
    
    # 工具调用
    'SQLToolClient',
    
    # 评估
    'Text2SQLEvaluator',
    'EvaluationResult',
    'EvaluationMetrics',
    'SQLExtractor',
    
    # 结果分析
    'ResultAnalyzer',
]

# 包级别的配置
DEFAULT_CONFIG = {
    'max_turns': 6,
    'concurrent_requests': 5,
    'conversation_timeout': 300,
    'sql_timeout': 30,
    'max_result_chars': 9000,
    'max_result_rows': 50,
} 