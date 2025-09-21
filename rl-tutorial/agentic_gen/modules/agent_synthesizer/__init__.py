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
Agent Synthesis Module
Responsible for generating diverse agent configurations based on existing tool libraries
"""

from .tool_graph import ToolGraph
from .tool_combination_generator import ToolCombinationGenerator
from .agent_config_generator import AgentConfigGenerator

from core.base_module import BaseModule
from typing import Dict, Any, List
import logging


class AgentSynthesizerModule(BaseModule):
    """Agent synthesis module main class"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize agent synthesis module
        
        Args:
            config: Module configuration
            logger: Logger
        """
        super().__init__(config, logger)
        
        # Initialize sub-components
        self.tool_combination_generator = None
        self.agent_config_generator = None
    
    def _setup(self):
        """Setup module components"""
        self.tool_combination_generator = ToolCombinationGenerator(self.logger)
        self.agent_config_generator = AgentConfigGenerator(self.logger)
        
        # Initialize sub-components
        self.tool_combination_generator.initialize()
        self.agent_config_generator.initialize()
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process agent synthesis
        
        Args:
            input_data: Input data, including tools and other information
            **kwargs: Other parameters
            
        Returns:
            Generated agent configuration data
        """
        try:
            tools = input_data.get('tools', [])
            target_agent_count = input_data.get('target_agent_count', 1000)
            
            if not tools:
                raise ValueError("No tools provided for agent synthesis")
            
            self.logger.info(f"Starting agent synthesis with {len(tools)} tools, target: {target_agent_count} agents")
            
            # 1. Generate tool combinations (based on intra-scenario similarity graph and random walk)
            self.logger.info("Generating tool combinations using graph-based random walk...")
            tool_combinations = self.tool_combination_generator.process({
                'tools': tools,
                'target_count': target_agent_count
            })
            
            # 2. Create tool data mapping
            tools_data = {tool['id']: tool for tool in tools}
            
            # 3. Generate agent configurations
            agents = self.agent_config_generator.process({
                'tool_combinations': tool_combinations,
                'tools_data': tools_data
            })
            
            self.logger.info(f"Successfully generated {len(agents)} agent configurations")
            
            return {
                'agents': agents,
                'tool_combinations': tool_combinations,
                'stats': {
                    'agent_count': len(agents),
                    'tool_combinations_count': len(tool_combinations),
                    'avg_tools_per_agent': sum(len(combo['tool_ids']) for combo in tool_combinations) / len(tool_combinations) if tool_combinations else 0,
                    'unique_tools_used': len(set(tool_id for combo in tool_combinations for tool_id in combo['tool_ids'])),
                    'agent_generation_success_rate': len(agents) / len(tool_combinations) if tool_combinations else 0,
                    'generation_method': 'fixed_template_simple'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Agent synthesis failed: {e}")
            raise


# Configuration constants
DEFAULT_TOOL_COUNT_RANGE = (3, 6)
DEFAULT_AGENT_COUNT = 1000
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_EMBEDDING_DIMENSIONS = 256

__all__ = [
    'AgentSynthesizerModule',
    'ToolGraph',
    'ToolCombinationGenerator',
    'AgentConfigGenerator',
    'DEFAULT_TOOL_COUNT_RANGE',
    'DEFAULT_AGENT_COUNT', 
    'DEFAULT_SIMILARITY_THRESHOLD',
    'DEFAULT_EMBEDDING_DIMENSIONS'
]