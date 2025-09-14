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
Tool Combination Generator
Builds graphs based on tool similarity within scenarios, generates tool combinations through random walks
"""

import random
from typing import Dict, Any, List, Tuple, Set
import logging
from datetime import datetime
from collections import defaultdict

from core.base_module import BaseModule
from core.exceptions import AgentDataGenException
from utils.data_processor import DataProcessor
from .tool_graph import ToolGraph


class ToolCombinationGenerator(BaseModule):
    """Tool Combination Generator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize tool combination generator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.data_processor = None
        self.tool_graph = None
        
        # Configuration parameters
        self.min_tools_per_agent = 3
        self.max_tools_per_agent = 6
        self.target_agent_count = 1000
        
        # Scenario distribution configuration
        self.scenario_sampling_weights = {}  # scenario_id -> weight
        self.max_agents_per_scenario = 50   # Maximum number of agents generated per scenario
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize tool graph
        self.tool_graph = ToolGraph(logger=self.logger)
        self.tool_graph.initialize()
        
        # 从配置读取参数
        agent_config = settings.GENERATION_CONFIG.get('agents', {})
        self.target_agent_count = agent_config.get('target_count', 1000)
        
        tools_per_agent = agent_config.get('tools_per_agent', {})
        self.min_tools_per_agent = tools_per_agent.get('min', 3)
        self.max_tools_per_agent = tools_per_agent.get('max', 6)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate tool combinations
        
        Args:
            input_data: Dictionary containing tool data
            **kwargs: Other parameters
            
        Returns:
            List of tool combinations
        """
        try:
            tools = input_data.get('tools', [])
            target_count = input_data.get('target_count', self.target_agent_count)
            
            if not tools:
                raise AgentDataGenException("No tools provided")
            
            self.logger.info(f"Generating {target_count} tool combinations from {len(tools)} tools")
            
            scenario_groups = self._group_tools_by_scenario(tools)
            self.logger.info(f"Found {len(scenario_groups)} scenario groups")
            
            scenario_graphs = self._build_scenario_graphs(scenario_groups)
            
            target_count_per_scenario = target_count // len(scenario_graphs)
            combinations = self._generate_combinations_from_scenarios(
                scenario_graphs, target_count_per_scenario
            )
            
            unique_combinations = self._deduplicate_combinations(combinations)
            
            self.logger.info(f"Generated {len(unique_combinations)} unique tool combinations")
            return unique_combinations
            
        except Exception as e:
            self.logger.error(f"Tool combination generation failed: {e}")
            raise AgentDataGenException(f"Failed to generate tool combinations: {e}")
    
    def _group_tools_by_scenario(self, tools: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tools by scenario"""
        scenario_groups = defaultdict(list)
        no_scenario_tools = []
        
        for tool in tools:
            scenario_ids = tool.get('scenario_ids', [])
            
            if scenario_ids:
                primary_scenario = scenario_ids[0]
                scenario_groups[primary_scenario].append(tool)
                
            else:
                no_scenario_tools.append(tool)
        
        filtered_groups = {
            scenario_id: tools_list 
            for scenario_id, tools_list in scenario_groups.items()
            if len(tools_list) >= self.min_tools_per_agent
        }
        
        self.logger.info(f"Scenario grouping: {len(filtered_groups)} valid scenarios")
        for scenario_id, tools_list in filtered_groups.items():
            self.logger.debug(f"  {scenario_id}: {len(tools_list)} tools")
        
        return filtered_groups
    
    def _build_scenario_graphs(self, scenario_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, ToolGraph]:
        """Build tool graph for each scenario"""
        scenario_graphs = {}
        
        for scenario_id, tools_list in scenario_groups.items():
            try:
                # Create new tool graph instance
                graph = ToolGraph(logger=self.logger)
                graph.initialize()
                
                graph.process({'tools': tools_list})
                
                scenario_graphs[scenario_id] = graph
                
                self.logger.debug(f"Built graph for scenario {scenario_id}: "
                                f"{graph.graph.number_of_nodes()} nodes, "
                                f"{graph.graph.number_of_edges()} edges")
                
            except Exception as e:
                self.logger.error(f"Failed to build graph for scenario {scenario_id}: {e}")
                continue
        
        return scenario_graphs
    
    def _generate_combinations_from_scenarios(self, scenario_graphs: Dict[str, ToolGraph], 
                                            target_count: int) -> List[Dict[str, Any]]:
        combinations = []
        
        for scenario_id, graph in scenario_graphs.items():
            target_for_scenario = target_count
                        
            scenario_combinations = self._generate_combinations_for_scenario(
                scenario_id, graph, target_for_scenario
            )
            
            combinations.extend(scenario_combinations)
        
        return combinations
    
    def _generate_combinations_for_scenario(self, scenario_id: str, graph: ToolGraph, 
                                          target_count: int) -> List[Dict[str, Any]]:
        """Generate tool combinations for a single scenario"""
        combinations = []
        available_tools = list(graph.graph.nodes())
        
        if len(available_tools) < self.min_tools_per_agent:
            self.logger.warning(f"Scenario {scenario_id} has too few tools: {len(available_tools)}")
            return combinations
            
        combinations = self._generate_random_walk_combinations(
            scenario_id, graph, target_count
        )
        
        return combinations[:target_count]
    
    def _generate_random_walk_combinations(self, scenario_id: str, graph: ToolGraph, 
                                         count: int) -> List[Dict[str, Any]]:
        """Generate combinations using random walk"""
        combinations = []
        available_tools = list(graph.graph.nodes())
        
        for i in range(count):
            start_tool = random.choice(available_tools)
            
            combo_size = random.randint(self.min_tools_per_agent, self.max_tools_per_agent)
            
            selected_tools = graph.random_walk_selection(start_tool, combo_size - 1)
            
            if start_tool not in selected_tools:
                selected_tools.insert(0, start_tool)
            
            selected_tools = selected_tools[:combo_size]
            
            while len(selected_tools) < self.min_tools_per_agent:
                remaining_tools = [t for t in available_tools if t not in selected_tools]
                if not remaining_tools:
                    break
                selected_tools.append(random.choice(remaining_tools))
            
            if len(selected_tools) >= self.min_tools_per_agent:
                combination = self._create_combination_record(
                    scenario_id, selected_tools, 'random_walk', start_tool
                )
                combinations.append(combination)
        
        return combinations
    
    def _create_combination_record(self, scenario_id: str, tool_ids: List[str], 
                                 method: str, start_tool: str) -> Dict[str, Any]:
        """创建组合记录"""
        combination_id = self.data_processor.generate_id('combination', {
            'scenario': scenario_id,
            'tools': sorted(tool_ids),
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'id': combination_id,
            'scenario_id': scenario_id,
            'tool_ids': tool_ids,
            'generation_method': method,
            'start_tool': start_tool,
            'tool_count': len(tool_ids),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'scenario_id': scenario_id,
                'generation_strategy': method
            }
        }
    
    def _deduplicate_combinations(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去除重复的工具组合"""
        seen_signatures = set()
        unique_combinations = []
        
        for combo in combinations:
            tool_signature = tuple(sorted(combo['tool_ids']))
            
            if tool_signature not in seen_signatures:
                seen_signatures.add(tool_signature)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def get_combination_stats(self, combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取组合统计信息"""
        if not combinations:
            return {}
        
        from collections import Counter
        
        total_combinations = len(combinations)
        tool_counts = [len(combo['tool_ids']) for combo in combinations]
        
        methods = [combo.get('generation_method', '') for combo in combinations]
        method_dist = Counter(methods)
        
        scenarios = [combo.get('scenario_id', '') for combo in combinations]
        scenario_dist = Counter(scenarios)
        
        all_tools = []
        for combo in combinations:
            all_tools.extend(combo['tool_ids'])
        tool_usage = Counter(all_tools)
        
        return {
            'total_combinations': total_combinations,
            'tool_count_distribution': {
                'min': min(tool_counts),
                'max': max(tool_counts),
                'avg': sum(tool_counts) / len(tool_counts),
                'distribution': dict(Counter(tool_counts))
            },
            'generation_method_distribution': dict(method_dist),
            'scenario_distribution': dict(scenario_dist.most_common(10)),
            'tool_usage_stats': {
                'unique_tools_used': len(tool_usage),
                'most_used_tools': dict(tool_usage.most_common(10)),
                'avg_usage_per_tool': sum(tool_usage.values()) / len(tool_usage)
            }
        }
