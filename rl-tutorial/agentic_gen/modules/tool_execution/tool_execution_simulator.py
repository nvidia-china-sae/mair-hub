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

"""
Tool Execution Simulator
Responsible for coordinating tool call parsing, execution, and state management
"""

import json
import re
from typing import Dict, Any, List
import logging

from core.base_module import BaseModule
from core.exceptions import AgentDataGenException
from .execution_engine import ExecutionEngine


class ToolExecutionSimulator(BaseModule):
    """Tool Execution Simulator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize tool execution simulator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.execution_engine = None
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize execution engine
        engine_config = settings.SIMULATOR_CONFIG
        self.execution_engine = ExecutionEngine(engine_config, self.logger)
        self.execution_engine.initialize()
    
    def initialize_tools(self, tools_info: Dict[str, Any]):
        """
        Initialize tool information
        
        Args:
            tools_info: Tool information dictionary
        """
        if self.execution_engine:
            self.execution_engine.register_tools(tools_info)
            self.logger.info(f"Initialized {len(tools_info)} tools for execution")
    
    def execute_agent_message(self, agent_message: str) -> List[Dict[str, Any]]:
        """
        Execute tool calls in agent message
        
        Args:
            agent_message: Agent message
            
        Returns:
            List of tool execution results
        """
        try:
            # Extract tool calls
            tool_calls = self._extract_tool_calls(agent_message)
            
            if not tool_calls:
                return []
            
            # Execute tool calls
            execution_data = {
                'tool_calls': tool_calls,
            }
            
            execution_results = self.execution_engine.process(execution_data)
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            return [{
                'tool_name': 'error',
                'status': 'failure',
                'message': f"Execution error: {e}",
                'result': None
            }]
    
    def _extract_tool_calls(self, agent_message: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from agent message.

        Args:
            agent_message: The message from the agent.

        Returns:
            List of tool calls, each as a dict {'name': str, 'arguments': dict}
        """
        import re
        import json

        def is_valid_tool_call_json(json_str: str) -> bool:
            try:
                parsed_json = json.loads(json_str)
                if not isinstance(parsed_json, dict):
                    return False
                if 'name' not in parsed_json:
                    return False
                if not isinstance(parsed_json['name'], str) or not parsed_json['name'].strip():
                    return False
                if 'arguments' in parsed_json and not isinstance(parsed_json['arguments'], dict):
                    return False
                return True
            except (json.JSONDecodeError, TypeError, KeyError):
                return False

        try:
            tool_calls = []
            processed_json_strings = set()

            # 1. Extract ```json ... ``` code blocks
            json_code_pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(json_code_pattern, agent_message, re.DOTALL)
            for match in matches:
                json_content = match.strip()
                if json_content in processed_json_strings:
                    continue
                processed_json_strings.add(json_content)
                try:
                    parsed_json = json.loads(json_content)
                    if isinstance(parsed_json, dict) and is_valid_tool_call_json(json_content):
                        tool_calls.append(parsed_json)
                    elif isinstance(parsed_json, list):
                        for item in parsed_json:
                            if isinstance(item, dict) and 'name' in item and is_valid_tool_call_json(json.dumps(item)):
                                tool_calls.append(item)
                except Exception:
                    continue

            # 2. Extract ``` ... ``` code blocks (no language specified)
            code_block_pattern = r'```\s*(.*?)\s*```'
            matches_code = re.findall(code_block_pattern, agent_message, re.DOTALL)
            for match in matches_code:
                code_content = match.strip()
                if code_content in processed_json_strings:
                    continue
                processed_json_strings.add(code_content)
                try:
                    parsed_json = json.loads(code_content)
                    if isinstance(parsed_json, dict) and is_valid_tool_call_json(code_content):
                        tool_calls.append(parsed_json)
                    elif isinstance(parsed_json, list):
                        for item in parsed_json:
                            if isinstance(item, dict) and 'name' in item and is_valid_tool_call_json(json.dumps(item)):
                                tool_calls.append(item)
                except Exception:
                    continue

            # 3. Remove all processed code blocks from message
            remaining_message = agent_message
            for match in matches:
                json_block = f"```json{match}```"
                remaining_message = remaining_message.replace(json_block, " ")
            for match in matches_code:
                code_block = f"```{match}```"
                remaining_message = remaining_message.replace(code_block, " ")

            # 4. Extract JSON objects {...}
            json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.finditer(json_object_pattern, remaining_message)
            for match in json_matches:
                json_str = match.group(0)
                if json_str in processed_json_strings:
                    continue
                processed_json_strings.add(json_str)
                if is_valid_tool_call_json(json_str):
                    try:
                        parsed_json = json.loads(json_str)
                        tool_calls.append(parsed_json)
                    except Exception:
                        continue

            # 5. Extract JSON arrays [...]
            json_array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
            array_matches = re.finditer(json_array_pattern, remaining_message)
            for match in array_matches:
                json_str = match.group(0)
                if json_str in processed_json_strings:
                    continue
                processed_json_strings.add(json_str)
                try:
                    parsed_json = json.loads(json_str)
                    if isinstance(parsed_json, list):
                        for item in parsed_json:
                            if isinstance(item, dict) and 'name' in item and is_valid_tool_call_json(json.dumps(item)):
                                tool_calls.append(item)
                except Exception:
                    continue

            # 6. Try to parse the whole message (after removing code blocks) as JSON (single line, fallback)
            cleaned_content = re.sub(r'\s+', ' ', remaining_message.strip())
            if cleaned_content not in processed_json_strings and is_valid_tool_call_json(cleaned_content):
                try:
                    parsed_json = json.loads(cleaned_content)
                    if isinstance(parsed_json, dict):
                        tool_calls.append(parsed_json)
                    elif isinstance(parsed_json, list):
                        for item in parsed_json:
                            if isinstance(item, dict) and 'name' in item and is_valid_tool_call_json(json.dumps(item)):
                                tool_calls.append(item)
                except Exception:
                    pass

            return tool_calls

        except Exception as e:
            self.logger.error(f"Failed to extract tool calls from message: {e}")
            return []
    
    def reset_execution_state(self):
        """Reset execution state"""
        if self.execution_engine:
            self.execution_engine.reset_execution_state()
            self.logger.info("Execution state reset")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_engine:
            return {}
        
        execution_state = self.execution_engine.get_execution_state()
        tool_usage = execution_state.get('tool_usage_count', {})
        
        return {
            'tool_usage_distribution': tool_usage,
            'execution_engine_state_size': len(str(execution_state)),
            'total_executions': sum(tool_usage.values()) if tool_usage else 0
        }