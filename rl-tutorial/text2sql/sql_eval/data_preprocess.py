#!/usr/bin/env python3
"""
数据预处理和Prompt构造模块

该模块负责：
1. 数据加载和预处理
2. Prompt构造
3. 数据库schema注入
4. 支持多种数据源格式
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器，负责处理不同数据源的数据库路径和schema获取"""
    
    def __init__(self, db_root_path: str):
        """
        初始化数据库管理器
        
        Args:
            db_root_path: 数据库根路径
        """
        self.db_root_path = Path(db_root_path)
        if not self.db_root_path.exists():
            raise ValueError(f"Database root path does not exist: {db_root_path}")
    
    def get_db_path(self, db_id: str, data_source: str) -> Path:
        """
        根据数据源和数据库ID获取数据库文件路径
        
        Args:
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            数据库文件路径
        """
        if data_source == 'synsql':
            return self.db_root_path / "SynSQL-2.5M" / "databases" / db_id / f"{db_id}.sqlite"
        elif data_source == 'spider':
            return self.db_root_path / "spider" / "test_database" / db_id / f"{db_id}.sqlite"
        elif data_source == 'bird':
            return self.db_root_path / "bird" / "train" / "train_databases" / db_id / f"{db_id}.sqlite"
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def get_database_schema(self, db_id: str, data_source: str) -> str:
        """
        获取数据库的schema信息
        
        Args:
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            数据库schema字符串
        """
        # 获取schema.sql文件路径
        schema_path = self.get_schema_path(db_id, data_source)
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        try:
            # 读取schema.sql文件内容
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_content = f.read()
            
            # 提取CREATE TABLE语句
            create_table_statements = self.extract_create_table_statements(schema_content)
            
            return "\n\n".join(create_table_statements)
            
        except Exception as e:
            logger.error(f"Error getting database schema for {db_id}: {e}")
            raise
    
    def get_schema_path(self, db_id: str, data_source: str) -> Path:
        """
        根据数据源和数据库ID获取schema.sql文件路径
        
        Args:
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            schema.sql文件路径
        """
        if data_source == 'synsql':
            return self.db_root_path / "SynSQL-2.5M" / "databases" / db_id / "schema.sql"
        elif data_source == 'spider':
            return self.db_root_path / "test_database" / db_id / "schema.sql"
        elif data_source == 'bird':
            return self.db_root_path / "bird" / "train" / "train_databases" / db_id / "schema.sql"
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def extract_create_table_statements(self, schema_content: str) -> List[str]:
        """
        从schema内容中提取CREATE TABLE语句
        
        Args:
            schema_content: schema文件内容
            
        Returns:
            CREATE TABLE语句列表
        """
        import re
        
        # 使用正则表达式匹配CREATE TABLE语句
        # 匹配模式：CREATE TABLE ... ; (包括多行)
        # 使用re.IGNORECASE忽略大小写，re.DOTALL让.匹配换行符
        pattern = r'create\s+table\s+[^;]+;'
        
        matches = re.findall(pattern, schema_content, re.IGNORECASE | re.DOTALL)
        
        # 清理匹配的语句
        create_table_statements = []
        for match in matches:
            # 保持原始的多行格式，但规范化空格
            formatted_statement = self.format_create_table_statement(match.strip())
            create_table_statements.append(formatted_statement)
        
        return create_table_statements
    
    def format_create_table_statement(self, statement: str) -> str:
        """
        格式化CREATE TABLE语句
        
        Args:
            statement: 原始CREATE TABLE语句
            
        Returns:
            格式化后的语句
        """
        # 简单的格式化：保持原有的换行和缩进结构
        lines = statement.split('\n')
        formatted_lines = []
        
        for line in lines:
            # 去除行首尾空白，但保持基本结构
            cleaned_line = line.strip()
            if cleaned_line:
                formatted_lines.append(cleaned_line)
        
        return '\n'.join(formatted_lines)


class PromptBuilder:
    """Prompt构造器，负责根据数据构造完整的对话prompt"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        初始化Prompt构造器
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
    
    def build_system_prompt(self) -> str:
        """
        构造系统prompt
        
        Returns:
            系统prompt字符串
        """
        return "You are a helpful assistant."
    
    def build_user_prompt(self, question: str, db_schema: str, external_knowledge: str = "") -> str:
        """
        构造用户prompt
        
        Args:
            question: 用户问题
            db_schema: 数据库schema
            external_knowledge: 外部知识（可选）
            
        Returns:
            用户prompt字符串
        """
        prompt_template = (
            """You are a data science expert. Your task is to understand database schemas """
            """and generate valid SQL queries to answer natural language questions using SQLite database engine. """
            """You must conduct reasoning inside <think> and </think> blocks every time you get new information. """
            """After reasoning, you need to explore or verify database information, you can call a SQL execution """
            """tool by <tool_call> execute_sql </tool_call> and it will return the query results between """
            """<tool_response> and </tool_response>. You can call SQL execution tool 6 times to explore """
            """the database structure and data. If you find no further exploration is needed, you MUST return """
            """your final SQL query enclosed within the <answer> </answer> tags."""
        )
        DEFAULT_USER_CONTENT = (
            "Remember, you should only output the SQL query itself inside the <answer> </answer> tags, not the results produced by executing the query."
        )

        user_content = (

            "{db_details}:{db_schema} \n" 

            "{external_knowledge}:;\n"

            "{question}: {question_value}"
        )

        prompt_template = prompt_template + "\n" +  user_content + "\n" + DEFAULT_USER_CONTENT
        
        return prompt_template.format(
            db_details="{db_details}",
            db_schema=db_schema,
            external_knowledge="{external_knowledge}",
            external_knowledge_value=external_knowledge,
            question="{question}",
            question_value=question
        )
    
    def build_messages(self, question: str, db_id: str, data_source: str, 
                      external_knowledge: str = "") -> List[Dict[str, str]]:
        """
        构造完整的对话消息
        
        Args:
            question: 用户问题
            db_id: 数据库ID
            data_source: 数据源类型
            external_knowledge: 外部知识（可选）
            
        Returns:
            消息列表
        """
        # 获取数据库schema
        db_schema = self.db_manager.get_database_schema(db_id, data_source)
        
        # 构造prompt
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(question, db_schema, external_knowledge)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        获取工具schema定义
        
        Returns:
            工具schema字典
        """
        return {
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


class DataProcessor:
    """数据处理器，负责加载和预处理评估数据"""
    
    def __init__(self, db_manager: DatabaseManager, prompt_builder: PromptBuilder):
        """
        初始化数据处理器
        
        Args:
            db_manager: 数据库管理器实例
            prompt_builder: Prompt构造器实例
        """
        self.db_manager = db_manager
        self.prompt_builder = prompt_builder
    
    def load_dataset(self, dataset_path: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载评估数据集
        
        Args:
            dataset_path: 数据集文件路径
            sample_size: 采样大小，None表示使用全部数据
            
        Returns:
            处理后的数据列表
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # 根据文件扩展名选择加载方式
        if dataset_path.suffix == '.parquet':
            df = pd.read_parquet(dataset_path)
        elif dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif dataset_path.suffix == '.jsonl':
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
        # 采样
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            # df = df.head(sample_size)
        
        # 转换为列表格式
        samples = []
        for idx, row in df.iterrows():
            # 处理不同的数据格式
            # 如果存在'query'字段，说明是Spider格式的数据
            if 'query' in row and pd.notna(row['query']):
                ground_truth_sql = row['query']
                data_source = row.get('data_source', 'spider')  # Spider数据默认使用spider
            else:
                # 原有格式，sql字段直接是SQL语句
                ground_truth_sql = row['sql']
                data_source = row.get('data_source', 'synsql')
            
            sample = {
                'id': row.get('id', idx),
                'question': row['question'],
                'db_id': row['db_id'],
                'data_source': data_source,
                'ground_truth_sql': ground_truth_sql,
                'external_knowledge': row.get('external_knowledge', ''),
                'difficulty': row.get('difficulty', 'unknown')
            }
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
        return samples
    
    def prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备单个样本，构造对话消息和相关信息
        
        Args:
            sample: 原始样本数据
            
        Returns:
            准备好的样本数据
        """
        try:
            # 构造对话消息
            messages = self.prompt_builder.build_messages(
                question=sample['question'],
                db_id=sample['db_id'],
                data_source=sample['data_source'],
                external_knowledge=sample['external_knowledge']
            )
            
            # 获取数据库路径
            db_path = self.db_manager.get_db_path(sample['db_id'], sample['data_source'])
            
            # 获取工具schema
            tool_schema = self.prompt_builder.get_tool_schema()
            
            prepared_sample = sample.copy()
            prepared_sample.update({
                'messages': messages,
                'db_path': str(db_path),
                'tool_schema': tool_schema
            })
            
            return prepared_sample
            
        except Exception as e:
            print(e)
            raise
    
    def prepare_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量准备样本
        
        Args:
            samples: 原始样本列表
            
        Returns:
            准备好的样本列表
        """
        prepared_samples = []
        
        for sample in samples:
            try:
                prepared_sample = self.prepare_sample(sample)
                prepared_samples.append(prepared_sample)
            except Exception as e:
                logger.warning(f"Skipping sample {sample.get('id', 'unknown')} due to error: {e}")
                continue
        
        logger.info(f"Successfully prepared {len(prepared_samples)}/{len(samples)} samples")
        return prepared_samples


def test_data_processor():
    """测试数据处理器功能"""
    import tempfile
    import os
    
    # 创建测试数据
    test_data = [
        {
        "db_id": "soccer_3",
        "query": "SELECT count(*) FROM club",
        "query_toks": [
            "SELECT",
            "count",
            "(",
            "*",
            ")",
            "FROM",
            "club"
        ],
        "query_toks_no_value": [
            "select",
            "count",
            "(",
            "*",
            ")",
            "from",
            "club"
        ],
        "question": "How many clubs are there?",
        "question_toks": [
            "How",
            "many",
            "clubs",
            "are",
            "there",
            "?"
        ],
    }
    ]
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        # 测试数据加载（注意：这需要实际的数据库路径）
        db_root_path = "./spider_data"  # 测试路径
        if not os.path.exists(db_root_path):
            os.makedirs(db_root_path)
        
        db_manager = DatabaseManager(db_root_path)
        prompt_builder = PromptBuilder(db_manager)
        data_processor = DataProcessor(db_manager, prompt_builder)
        
        # 加载数据
        samples = data_processor.load_dataset(temp_file)
        print(f"Loaded {len(samples)} samples")

        prepare_sample = data_processor.prepare_sample(samples[0])
        print(prepare_sample.keys())
        
        # 打印第一个样本
        if samples:
            print("Sample data:")
            print(json.dumps(samples[0], indent=2, ensure_ascii=False))
        
    finally:
        # 清理临时文件
        os.unlink(temp_file)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_data_processor() 