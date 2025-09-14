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
Agent Simulator
Responsible for simulating agent behavior and decision-making
"""

import json
import re
from typing import Dict, Any, List
import logging


import sys
from pathlib import Path

from core.base_module import BaseModule
from core.models import AgentConfig
from core.exceptions import AgentDataGenException
from utils.llm_client import LLMClient
from config.prompts.agent_prompts import AgentPrompts


class AgentSimulator(BaseModule):
    """Agent Simulator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize agent simulator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.llm_client = None
        self.prompts = AgentPrompts()
        
        # Current state
        self.current_agent_config = None
        self.tools_info = {}
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize LLM client
        llm_config = settings.get_llm_config()
        llm_config['provider'] = settings.DEFAULT_LLM_PROVIDER
        self.llm_client = LLMClient(llm_config, self.logger)
    
    def initialize_for_agent(self, agent_config: AgentConfig, tools_info: Dict[str, Any]):
        """
        Initialize simulator for agent
        
        Args:
            agent_config: Agent configuration
            tools_info: Tools information
        """
        self.current_agent_config = agent_config
        self.tools_info = tools_info
        self.logger.info(f"Initialized agent simulator for agent {agent_config.id}")
    
    def respond(self, conversation_history: str) -> Dict[str, Any]:
        """
        Generate agent response based on conversation history
        Refers to APIAgent_turn implementation in other_project_fils
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            Response dictionary containing sender, recipient, message
        """
        try:
            if not self.current_agent_config:
                raise AgentDataGenException("No agent configuration set")
            
            
            system_prompt = self.current_agent_config.system_prompt
            # Build user prompt
            user_prompt = self.prompts.AGENT_USER.format(conversation_history=conversation_history)
            # Call LLM to generate response
            response = self.llm_client.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            response_content = response.content.strip()
            # Build current message
            current_message = {"sender": "agent"}
            
            # Determine if it contains tool call
            # Refer to other_project_fils logic
            if self._contains_tool_call(response_content):
                current_message["recipient"] = "execution"
                current_message["message"] = response_content
            else:
                current_message["recipient"] = "user"
                current_message["message"] = response_content
            
            return current_message
            
        except Exception as e:
            self.logger.error(f"Failed to generate agent response: {e}")
            return {
                "sender": "agent",
                "recipient": "user",
                "message": "Sorry, I encountered some issues, please try again later."
            }
    
    def _contains_tool_call(self, response_content: str) -> bool:
        """
        Determine if response contains tool call
        Supports multiple formats: ```json ... ```, ``` ... ```, plain JSON objects
        
        Args:
            response_content: Response content
            
        Returns:
            Whether it contains tool call
        """
        try:
            # 1. First try to parse ```json ... ``` format
            json_code_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_code_pattern, response_content, re.DOTALL)
            
            if match:
                json_content = match.group(1).strip()
                if self._is_valid_tool_call_json(json_content):
                    return True
            
            # 2. Try to parse ``` ... ``` format (no language specified)
            code_block_pattern = r'```\s*(.*?)\s*```'
            match = re.search(code_block_pattern, response_content, re.DOTALL)
            
            if match:
                code_content = match.group(1).strip()
                if self._is_valid_tool_call_json(code_content):
                    return True
            
            # 3. Try to extract plain JSON objects
            # Improved regex to better handle nested JSON
            json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.finditer(json_object_pattern, response_content)
            
            for match in json_matches:
                json_str = match.group(0)
                if self._is_valid_tool_call_json(json_str):
                    return True
            
            # 4. Try to extract single-line JSON (handle possible line breaks)
            # Remove line breaks and extra spaces then try to parse
            cleaned_content = re.sub(r'\s+', ' ', response_content.strip())
            if self._is_valid_tool_call_json(cleaned_content):
                return True
                
            return False

        except Exception as e:
            self.logger.error(f"Failed to check tool call: {e}")
            return False
    
    def _is_valid_tool_call_json(self, json_str: str) -> bool:
        """
        Validate if JSON string is a valid tool call format
        
        Args:
            json_str: JSON string
            
        Returns:
            Whether it is a valid tool call
        """
        try:
            parsed_json = json.loads(json_str)
            
            # Check if it is a dictionary type
            if not isinstance(parsed_json, dict):
                return False
            
            # Check if it contains necessary fields
            if 'name' not in parsed_json:
                return False
            
            # Check if name field is a string and not empty
            if not isinstance(parsed_json['name'], str) or not parsed_json['name'].strip():
                return False
            
            # Check arguments field (if exists)
            if 'arguments' in parsed_json:
                if not isinstance(parsed_json['arguments'], dict):
                    return False
            
            return True
            
        except (json.JSONDecodeError, TypeError, KeyError):
            return False

if __name__ == "__main__":
    # Note spelling: AgentSimulator
    from modules.agent_simulator.agent_simulator import AgentSimulator

    # Create simulator instance (logger can be None or custom here)
    simulator = AgentSimulator(logger=None)

    # Test string
    test_str = '{"name": "tag_department_codes", "arguments": {"department_map": {"user_7xK2m": "Sales", "user_9pL4q": "Marketing", "user_2wN8r": "Product", "user_5hJ1k": "Sales", "user_3dF6v": "Marketing"}}}'
    result = simulator._contains_tool_call(test_str)
    test_str = '{"name": "generate_dept_invoice", "arguments": {"session_token": "sess_admin_aXyZ9", "date": "2024-05-31"}}'
    # Call _contains_tool_call method
    result = simulator._contains_tool_call(test_str)

    print(f"Tool call detected: {result}")