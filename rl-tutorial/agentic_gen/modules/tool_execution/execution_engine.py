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
Tool Execution Engine
Responsible for parsing and executing tool calls
"""

import json
import re
import ast
import random
import time
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from core.base_module import BaseModule
from core.exceptions import AgentDataGenException
from utils.llm_client import LLMClient
from utils.data_processor import DataProcessor
from config.prompts.execution_prompts import ExecutionPrompts


class ExecutionEngine(BaseModule):
    """Tool Execution Engine"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize execution engine
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.llm_client = None
        self.data_processor = None
        self.prompts = ExecutionPrompts()
        
        # Execution state
        self.execution_state = {}
        self.tools_registry = {}
        
        # Execution configuration
        self.success_rate = 0.85
        self.partial_failure_rate = 0.10
        self.complete_failure_rate = 0.05
        self.randomness_level = 0.1
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize LLM client
        llm_config = settings.get_llm_config()
        llm_config['provider'] = settings.DEFAULT_LLM_PROVIDER
        self.llm_client = LLMClient(llm_config, self.logger)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Update execution configuration
        simulator_config = settings.SIMULATOR_CONFIG
        self.success_rate = simulator_config.get('success_rate', 0.85)
        self.partial_failure_rate = simulator_config.get('partial_failure_rate', 0.10)
        self.complete_failure_rate = simulator_config.get('complete_failure_rate', 0.05)
        self.randomness_level = simulator_config.get('randomness_level', 0.1)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process tool execution request
        
        Args:
            input_data: Data containing tool call information
            **kwargs: Other parameters
            
        Returns:
            Execution results
        """
        try:
            tool_calls = input_data.get('tool_calls', [])
            
            if not tool_calls:
                return {'results': [], 'errors': ['No tool calls provided']}
            
            
            # Execute tool calls
            results = []
            errors = []
            
            for tool_call in tool_calls:
                try:
                    result = self.execute_tool_call(tool_call)
                    results.append(result)
                except Exception as e:
                    error_msg = f"Failed to execute tool call: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Execution engine process failed: {e}")
            raise AgentDataGenException(f"Execution failed: {e}")
    
    def execute_tool_call(self, tool_call: str) -> Dict[str, Any]:
        """
        Execute single tool call
        
        Args:
            tool_call: Tool call information
            
        Returns:
            Execution result
        """
        try:
            # Parse tool call
            tool_name = tool_call.get('name')
            parameters = tool_call.get('arguments', {})
            # Get tool information
            tool_info = self.tools_registry.get(tool_name)
            if not tool_info:
                return self._create_error_result(
                    tool_name, 
                    f"Tool '{tool_name}' not found in registry",
                    'tool_not_found'
                )
            
            # Validate parameters
            validation_result = self._validate_parameters(tool_info, parameters)
            if not validation_result['valid']:
                return self._create_error_result(
                    tool_name,
                    validation_result['error'],
                    'parameter_error'
                )
            
            # Determine execution result type
            execution_type = self._determine_execution_type()
            
            # Simulate execution
            result = self._simulate_execution(tool_call, tool_info, execution_type)
            
            # Update execution state
            self._update_execution_state(tool_name, parameters, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute tool call '{tool_call}': {e}")
            return self._create_error_result(
                'unknown',
                f"Execution error: {e}",
                'system_error'
            )

    def _validate_parameters(self, tool_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters"""
        try:
            tool_parameters = tool_info.get('parameters', [])
            required_params = [p['name'] for p in tool_parameters if p.get('required', True)]
            
            # Check required parameters
            missing_params = [param for param in required_params if param not in parameters]
            if missing_params:
                return {
                    'valid': False,
                    'error': f"Missing required parameters: {missing_params}"
                }
            
            # Check parameter types (simple validation)
            for param_info in tool_parameters:
                param_name = param_info['name']
                if param_name in parameters:
                    expected_type = param_info.get('type', 'string')
                    param_value = parameters[param_name]
                    
                    # Simple type checking
                    if expected_type == 'integer' and not isinstance(param_value, int):
                        try:
                            parameters[param_name] = int(param_value)
                        except (ValueError, TypeError):
                            return {
                                'valid': False,
                                'error': f"Parameter '{param_name}' should be an integer"
                            }
                    elif expected_type == 'number' and not isinstance(param_value, (int, float)):
                        try:
                            parameters[param_name] = float(param_value)
                        except (ValueError, TypeError):
                            return {
                                'valid': False,
                                'error': f"Parameter '{param_name}' should be a number"
                            }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Parameter validation error: {e}"
            }
    
    def _determine_execution_type(self) -> str:
        """Determine execution result type"""
        rand = random.random()
        
        if rand < self.success_rate:
            return 'success'
        elif rand < self.success_rate + self.partial_failure_rate:
            return 'partial_success'
        else:
            return 'failure'
    
    def _simulate_execution(self, tool_call: Dict[str, Any], tool_info: Dict[str, Any], execution_type: str) -> Dict[str, Any]:
        """Unified simulation execution method"""
        try:
            # Build tool call information
            tool_call_text = json.dumps(tool_call, ensure_ascii=False, indent=2)
            examples_text = json.dumps(tool_info.get('examples', []), ensure_ascii=False, indent=2)
            
            
            # Build prompt
            prompt = self.prompts.EXECUTION_RESULT_TEMPLATE.format(
                tool_call=tool_call_text,
                examples=examples_text,
                execution_type=execution_type,
                current_state=json.dumps(self.execution_state, ensure_ascii=False, indent=2)
            )
            
            # Call LLM to generate result
            response = self.llm_client.generate_completion(
                prompt=prompt,
                system_prompt=self.prompts.TOOL_EXECUTION_SYSTEM,
            )            
            # Parse LLM response
            try:
                result = self.llm_client.parse_json_response(response)
            except Exception as parse_error:
                self.logger.warning(f"Failed to parse LLM response: {parse_error}")
                return self._create_default_result(tool_name, parameters, execution_type)
            
            # Ensure result format is correct
            if 'status' not in result:
                result['status'] = execution_type if execution_type != 'partial_success' else 'success'
            if 'metadata' not in result:
                result['metadata'] = {}
            
            tool_name = tool_info.get('name', 'unknown')
            result['metadata'].update({
                'tool_name': tool_name,
                'timestamp': datetime.now().isoformat(),
                'execution_time': result.get('metadata', {}).get('execution_time', round(random.uniform(0.1, 2.0), 2)),
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to simulate execution: {e}")
            tool_name = tool_call.get('name', 'unknown')
            parameters = tool_call.get('arguments', {})
            return self._create_default_result(tool_name, parameters, execution_type)
    
    def _create_error_result(self, tool_name: str, error_message: str, error_type: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "status": "failure",
            "result": None,
            "message": error_message,
            "metadata": {
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
                "error_type": error_type,
                "execution_time": 0.0
            }
        }
    
    def _create_default_result(self, tool_name: str, parameters: Dict[str, Any], execution_type: str) -> Dict[str, Any]:
        """Create default result"""
        if execution_type == 'success':
            return {
                "status": "success",
                "result": f"Tool {tool_name} executed successfully with parameters: {parameters}",
                "message": "Operation completed successfully",
                "metadata": {
                    "tool_name": tool_name,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": round(random.uniform(0.1, 2.0), 2),
                    "execution_type": execution_type
                }
            }
        elif execution_type == 'partial_success':
            return {
                "status": "success",
                "result": f"Tool {tool_name} executed with partial success",
                "message": "Operation completed with warnings",
                "metadata": {
                    "tool_name": tool_name,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": round(random.uniform(0.1, 2.0), 2),
                    "execution_type": execution_type,
                    "warnings": ["Some optional parameters were missing or invalid"]
                }
            }
        else:  # failure
            return {
                "status": "failure",
                "result": None,
                "message": f"Tool {tool_name} execution failed",
                "metadata": {
                    "tool_name": tool_name,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": 0.0,
                    "execution_type": execution_type,
                    "error_type": "execution_error"
                }
            }
    
    def _update_execution_state(self, tool_name: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Update execution state"""
        try:
            # Record execution history
            if 'execution_history' not in self.execution_state:
                self.execution_state['execution_history'] = []
            
            execution_record = {
                'tool_name': tool_name,
                'parameters': parameters,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.execution_state['execution_history'].append(execution_record)

        except Exception as e:
            self.logger.error(f"Failed to update execution state: {e}")    

    def register_tools(self, tools_info: Dict[str, Any]):
        """Register tool information"""
        self.tools_registry.update(tools_info)
        self.logger.info(f"Registered {len(tools_info)} tools")