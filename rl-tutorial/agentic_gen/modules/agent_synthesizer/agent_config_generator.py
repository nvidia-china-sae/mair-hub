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
Agent Configuration Generator
Integrates tool combinations and prompts to generate complete agent configurations
"""

import random
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from core.base_module import BaseModule
from core.models import AgentConfig
from core.exceptions import AgentDataGenException
from utils.data_processor import DataProcessor
from utils.file_manager import FileManager


class AgentConfigGenerator(BaseModule):
    """Agent Configuration Generator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize agent configuration generator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.data_processor = None
        self.file_manager = None
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize file manager
        data_path = settings.get_data_path('agents')
        self.file_manager = FileManager(data_path, self.logger)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> List[AgentConfig]:
        """
        Generate agent configurations
        
        Args:
            input_data: Input data containing tool combinations
            **kwargs: Other parameters
            
        Returns:
            List of agent configurations
        """
        try:
            tool_combinations = input_data.get('tool_combinations', [])
            tools_data = input_data.get('tools_data', {})
            
            if not tool_combinations:
                raise AgentDataGenException("No tool combinations provided")
            
            self.logger.info(f"Generating agent configurations for {len(tool_combinations)} combinations")
            
            agents = []
            for i, combination in enumerate(tool_combinations):
                try:
                    agent = self._generate_agent_config(combination, tools_data)
                    if agent:
                        agents.append(agent)
                        
                except Exception as e:
                    self.logger.error(f"Failed to generate agent for combination {i}: {e}")
                    continue
            
            # Save agent configurations
            self._save_agent_configs(agents)
            
            self.logger.info(f"Successfully generated {len(agents)} agent configurations")
            return agents
            
        except Exception as e:
            self.logger.error(f"Agent configuration generation failed: {e}")
            raise AgentDataGenException(f"Failed to generate agent configurations: {e}")
    
    def _generate_agent_config(self, combination: Dict[str, Any], 
                                     tools_data: Dict[str, Any]) -> Optional[AgentConfig]:
        """Generate single agent configuration"""
        
        try:
            # 1. Generate agent ID
            agent_id = self.data_processor.generate_id('agent', combination)
            
            # 2. Get tool list
            tool_ids = combination.get('tool_ids', [])
            if not tool_ids:
                raise ValueError("No tools in combination")
            
            # 3. Generate system prompt
            system_prompt = self._build_system_prompt_with_tools(tool_ids, tools_data)
            # 4. Create AgentConfig object
            agent_config = AgentConfig(
                id=agent_id,
                system_prompt=system_prompt,
                tools=tool_ids,
            )
            
            return agent_config
            
        except Exception as e:
            self.logger.error(f"Failed to generate agent config: {e}")
            return None
    
    def _build_system_prompt_with_tools(self, tool_ids: List[str], tools_data: Dict[str, Any]) -> str:
        """Build system prompt containing tool list"""
        from config.prompts.agent_prompts import AgentPrompts
        
        # Get detailed tool information
        tools_info = []
        for tool_id in tool_ids:
            if tool_id in tools_data:
                tools_info.append(tools_data[tool_id])
        
        # Build tool list text
        tools_list = self._build_tools_list(tools_info)
        
        # Use fixed template
        prompts = AgentPrompts()
        return prompts.AGENT_SYSTEM.format(tools_list=tools_list)
    
    def _build_tools_list(self, tools_info: List[Dict[str, Any]]) -> str:
        """Build JSON format tool list"""
        import json
        
        tools_json_list = []
        
        for tool in tools_info:
            tool_name = tool.get('name', '')
            tool_desc = tool.get('description', '')
            parameters = tool.get('parameters', [])
            
            # Build JSON Schema format tool definition
            tool_json = {
                "name": tool_name,
                "description": tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Process parameters
            for param in parameters:
                param_name = param.get('name', '')
                param_type = param.get('type', 'string')
                param_desc = param.get('description', '')
                required = param.get('required', False)
                enum_values = param.get('enum', None)
                
                # Build parameter definition
                param_def = {
                    "type": param_type,
                    "description": param_desc
                }
                
                # Add enum values (if any)
                if enum_values:
                    param_def["enum"] = enum_values
                
                # Add to properties
                tool_json["parameters"]["properties"][param_name] = param_def
                
                # Add to required (if mandatory)
                if required:
                    tool_json["parameters"]["required"].append(param_name)
            
            # Format JSON and add to list
            tool_json_str = json.dumps(tool_json, ensure_ascii=False, indent=2)
            tools_json_list.append(tool_json_str)
        
        return '\n\n'.join(tools_json_list)

    
    def _save_agent_configs(self, agents: List[AgentConfig]):
        """Save agent configurations"""
        try:
            # Convert to serializable format
            agents_data = []
            for agent in agents:
                agent_dict = {
                    'id': agent.id,
                    'system_prompt': agent.system_prompt,
                    'tools': agent.tools,
                    'created_at': agent.created_at.isoformat() if hasattr(agent, 'created_at') and agent.created_at else datetime.now().isoformat()
                }
                agents_data.append(agent_dict)
            
            # Save main file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agents_batch_{timestamp}.json"
            
            self.file_manager.save_json(agents_data, filename)
            
        except Exception as e:
            self.logger.error(f"Failed to save agent configs: {e}")
            raise AgentDataGenException(f"Failed to save agents: {e}")

    
