#!/usr/bin/env python3
"""
多轮对话管理器模块

该模块负责：
1. 管理与SGLang服务器的通信
2. 处理多轮对话状态
3. 处理工具调用请求
4. 结果解析和状态转换
5. 超时和异常处理
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from openai import AsyncOpenAI

try:
    from .tool_client import SQLToolClient
except ImportError:
    from tool_client import SQLToolClient

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """对话状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    TOOL_CALLING = "tool_calling"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ToolCall:
    """工具调用数据类"""
    id: str
    type: str
    function: Dict[str, Any]


@dataclass
class Message:
    """消息数据类"""
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ConversationManager:
    """多轮对话管理器，负责与SGLang服务器通信和对话状态管理"""
    
    def __init__(self, server_url: str, sql_tool_client: SQLToolClient, 
                 max_turns: int = 6, timeout: int = 300,
                 model_name: str = "qwen2.5-7b-instruct", temperature: float = 0.6,
                 max_tokens: int = 30000, stream: bool = False):
        """
        初始化对话管理器
        
        Args:
            server_url: SGLang服务器URL
            sql_tool_client: SQL工具客户端
            max_turns: 最大对话轮数
            timeout: 请求超时时间
            model_name: 模型名称
            temperature: 温度系数
            max_tokens: 最大token数
            stream: 是否启用流式输出
        """
        self.server_url = server_url.rstrip('/')
        self.sql_tool_client = sql_tool_client
        self.max_turns = max_turns
        self.timeout = timeout
        
        # 模型参数
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        
        # 对话状态
        self.state = ConversationState.PENDING
        self.messages: List[Message] = []
        self.current_turn = 0
        self.conversation_history: List[Dict[str, Any]] = []
        
        # 工具相关
        self.tool_schema = self._get_tool_schema()
        
        # OpenAI 客户端
        self.client = None
    
    def _get_tool_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return [
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
        ]
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.client = AsyncOpenAI(
            base_url=f"{self.server_url}/v1",
            api_key="EMPTY",  # SGLang 不需要真实的 API key
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.close()
    
    def reset(self):
        """重置对话状态"""
        self.state = ConversationState.PENDING
        self.messages.clear()
        self.current_turn = 0
        self.conversation_history.clear()
    
    def add_message(self, role: str, content: str, tool_calls: Optional[List[ToolCall]] = None,
                   tool_call_id: Optional[str] = None):
        """添加消息到对话历史"""
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        )
        self.messages.append(message)
        
        # 记录到对话历史
        msg_dict = {
            "role": role,
            "content": content
        }
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in tool_calls
            ]
        if tool_call_id:
            msg_dict["tool_call_id"] = tool_call_id
        
        self.conversation_history.append(msg_dict)
    
    async def call_model(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        调用SGLang模型API
        
        Args:
            messages: 消息列表
            tools: 工具列表（可选）
            
        Returns:
            模型响应
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            # 构造请求参数
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": self.stream
            }
            
            if tools:
                request_params["tools"] = tools
            # print("messages:", messages)
            # 使用 OpenAI 客户端发送请求
            response = await self.client.chat.completions.create(**request_params)
            
            # SGLang 返回的是标准的 JSON 对象，直接转换为字典
            result = response.model_dump()
            
            return result
                
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Model request timeout after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            raise
    
    def parse_tool_calls_from_response(self, response: Dict[str, Any]) -> Tuple[str, List[ToolCall], str]:
        """
        从模型响应中解析工具调用，参考 sglang_rollout.py 的逻辑
        
        Args:
            response: 模型响应
            
        Returns:
            (content, tool_calls, finish_reason)
        """
        if "choices" not in response or not response["choices"]:
            raise ValueError("Invalid model response: no choices")
        
        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")
        
        tool_calls = []
        
        # 检查是否有 OpenAI 格式的工具调用
        if finish_reason == "tool_calls" and "tool_calls" in message:
            # OpenAI 格式的工具调用
            for tc_data in message["tool_calls"]:
                tool_call = ToolCall(
                    id=tc_data["id"],
                    type=tc_data["type"],
                    function=tc_data["function"]
                )
                tool_calls.append(tool_call)
        
        return content, tool_calls, finish_reason
    
    async def execute_tool_call(self, tool_call: ToolCall, db_id: str, data_source: str) -> Tuple[str, Dict[str, Any]]:
        """
        执行工具调用
        
        Args:
            tool_call: 工具调用对象
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            (工具响应, 执行指标)
        """
        if tool_call.function["name"] != "execute_sql":
            raise ValueError(f"Unsupported tool: {tool_call.function['name']}")
        
        try:
            # 解析参数 - 从 schema 中直接获取 sql_query
            if isinstance(tool_call.function["arguments"], str):
                arguments = json.loads(tool_call.function["arguments"])
            else:
                arguments = tool_call.function["arguments"]
            
            sql_query = arguments.get("sql_query", "")
            
            if not sql_query:
                return "Error: No SQL query provided", {"error": "No SQL query provided"}
            # print("sql query:", sql_query)
            # 执行SQL查询
            response_text, success, metrics = self.sql_tool_client.execute_sql(
                sql_query=sql_query,
                db_id=db_id,
                data_source=data_source
            )
            
            # 解析响应
            try:
                response_data = json.loads(response_text)
                result = response_data.get("result", "No result")
            except json.JSONDecodeError:
                result = response_text
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Error executing tool call: {e}")
            return f"Tool execution error: {str(e)}", {"error": str(e)}
    
    async def run_conversation(self, initial_messages: List[Dict[str, str]], 
                             db_id: str, data_source: str) -> Dict[str, Any]:
        """
        运行完整的多轮对话
        
        Args:
            initial_messages: 初始消息列表
            db_id: 数据库ID
            data_source: 数据源类型
            
        Returns:
            对话结果
        """
        self.reset()
        
        # 添加初始消息
        for msg in initial_messages:
            self.add_message(msg["role"], msg["content"])
        
        self.state = ConversationState.RUNNING
        tool_call_count = 0
        
        try:
            while self.current_turn < self.max_turns and self.state != ConversationState.COMPLETED:
                logger.info(f"Starting conversation turn {self.current_turn + 1}")
                
                # 准备消息列表
                messages_for_api = []
                for msg in self.messages:
                    msg_dict = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    if msg.tool_calls:
                        msg_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": tc.function
                            }
                            for tc in msg.tool_calls
                        ]
                    if msg.tool_call_id:
                        msg_dict["tool_call_id"] = msg.tool_call_id
                    
                    messages_for_api.append(msg_dict)
                
                # 调用模型
                tools = self.tool_schema
                response = await self.call_model(messages_for_api, tools)
                
                # 解析模型响应
                content, tool_calls, finish_reason = self.parse_tool_calls_from_response(response)
                
                logger.info(f"Model response - finish_reason: {finish_reason}, content: {content}")
                
                # 添加助手消息
                self.add_message("assistant", content, tool_calls if tool_calls else None)
                self.current_turn += 1
                
                # 如果有工具调用，执行工具并继续对话
                if tool_calls:
                    self.state = ConversationState.TOOL_CALLING
                    
                    for tool_call in tool_calls:
                        try:
                            # 执行工具调用
                            tool_response, tool_metrics = await self.execute_tool_call(
                                tool_call, db_id, data_source
                            )
                            
                            # 添加工具响应消息
                            self.add_message(
                                "tool",
                                tool_response,
                                tool_call_id=tool_call.id
                            )
                            
                            tool_call_count += 1
                            logger.info(f"Executed tool call {tool_call_count}: {tool_call.function['name']}")
                            
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            error_response = f"Tool execution failed: {str(e)}"
                            self.add_message(
                                "tool",
                                error_response,
                                tool_call_id=tool_call.id
                            )
                    
                    self.state = ConversationState.RUNNING
                
                # 检查是否应该结束对话
                elif finish_reason == "stop":
                    self.state = ConversationState.COMPLETED
                    break
                elif finish_reason == "length":
                    logger.warning("Model response was truncated due to length")
                    self.state = ConversationState.COMPLETED
                    break
            
            # 检查是否达到最大轮次但没有最终答案
            if self.current_turn >= self.max_turns and self.state != ConversationState.COMPLETED:
                logger.info(f"Max turns reached ({self.max_turns}). Checking for final answer.")
                
                # 查找最后一条助手消息
                last_assistant_message = None
                for msg in reversed(self.messages):
                    if msg.role == "assistant":
                        last_assistant_message = msg
                        break
                
                # 如果最后一条助手消息中没有 <answer> 标签，生成最终答案
                if last_assistant_message is not None and "<answer>" not in last_assistant_message.content:
                    logger.info(f"No <answer> tags found in last assistant message. Requesting final answer.")
                    
                    try:
                        # 添加最终提示消息
                        final_prompt = "You have reached the maximum number of interaction turns. Please provide your final sql query directly using the <answer></answer> tags."
                        self.add_message("user", final_prompt)
                        
                        # 准备消息列表
                        messages_for_api = []
                        for msg in self.messages:
                            msg_dict = {
                                "role": msg.role,
                                "content": msg.content
                            }
                            if msg.tool_calls:
                                msg_dict["tool_calls"] = [
                                    {
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": tc.function
                                    }
                                    for tc in msg.tool_calls
                                ]
                            if msg.tool_call_id:
                                msg_dict["tool_call_id"] = msg.tool_call_id
                            
                            messages_for_api.append(msg_dict)
                        
                        # 发送最终请求（不提供工具，强制生成文本答案）
                        final_response = await self.call_model(messages_for_api, tools=None)
                        
                        # 解析最终响应
                        final_content, final_tool_calls, final_finish_reason = self.parse_tool_calls_from_response(final_response)
                        
                        # 添加最终助手消息
                        self.add_message("assistant", final_content)
                        self.current_turn += 1
                        
                        logger.info(f"Generated final response with finish reason: {final_finish_reason}")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Final content: {final_content[:200]}...")
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate final response: {e}")
                        # 如果最终生成失败，继续使用原有结果
                
                self.state = ConversationState.COMPLETED

            # 构造最终结果
            result = {
                "success": True,
                "state": self.state.value,
                "turns": self.current_turn,
                "tool_calls": tool_call_count,
                "conversation_history": self.conversation_history,
                "final_response": self.messages[-1].content if self.messages else "",
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": tc.function
                            }
                            for tc in msg.tool_calls
                        ] if msg.tool_calls else None,
                        "tool_call_id": msg.tool_call_id
                    }
                    for msg in self.messages
                ]
            }
            # print("messages:", self.messages)
            logger.info(f"Conversation completed - turns: {self.current_turn}, tool_calls: {tool_call_count}")
            return result
            
        except Exception as e:
            logger.error(f"Conversation failed: {e}")
            self.state = ConversationState.ERROR
            
            return {
                "success": False,
                "error": str(e),
                "state": self.state.value,
                "turns": self.current_turn,
                "conversation_history": self.conversation_history,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content
                    }
                    for msg in self.messages
                ]
            }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "state": self.state.value,
            "turns": self.current_turn,
            "message_count": len(self.messages),
            "tool_calls": len([msg for msg in self.messages if msg.tool_calls]),
            "last_message": self.messages[-1].content if self.messages else None
        }


async def test_conversation_manager():
    """测试对话管理器功能"""
    import tempfile
    import os
    import json
    
    # 使用与 test_data_processor 相同的测试数据
    test_data = [
        {
        "db_id": "soccer_3",
        "query": "SELECT Manager ,  Captain FROM club",
        "question": "What are the managers and captains of clubs?",
    }
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    
    # 创建临时目录和文件
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {temp_dir}")
    
    try:
        # 导入必要的模块
        try:
            from .data_preprocess import DatabaseManager, PromptBuilder, DataProcessor
        except ImportError:
            from data_preprocess import DatabaseManager, PromptBuilder, DataProcessor
        
        db_root_path = "./spider_data"  # 测试路径
        
        
        # 创建数据库管理器、提示构建器和数据处理器
        db_manager = DatabaseManager(db_root_path)
        prompt_builder = PromptBuilder(db_manager)
        data_processor = DataProcessor(db_manager, prompt_builder)
        
        # 创建SQL工具客户端

        samples = data_processor.load_dataset(temp_file)
        print(f"Loaded {len(samples)} samples")
        # 准备测试样本
        prepared_sample = data_processor.prepare_sample(samples[0])
        
        print("Prepared sample keys:", list(prepared_sample.keys()))
        # print("messages:", prepared_sample['messages'])
        messages = prepared_sample['messages']
        sql_client = SQLToolClient(db_root_path)
        # 创建对话管理器
        server_url = "http://localhost:30000"  # 假设的服务器地址
        
        async with ConversationManager(server_url, sql_client, max_turns=6, timeout=3000) as manager:
            conversation_result = await manager.run_conversation(
                messages,
                prepared_sample['db_id'],
                'spider'
            )
            for message in conversation_result["conversation_history"]:
                print('role:', message['role'], 'content:', message['content'])
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    asyncio.run(test_conversation_manager()) 
