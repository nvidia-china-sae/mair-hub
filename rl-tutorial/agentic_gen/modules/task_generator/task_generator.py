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
Task Generation Module
Responsible for generating multi-turn conversation tasks and scoring criteria for agents
"""

import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.base_module import BaseModule
from core.models import Task, AgentConfig, DifficultyLevel
from core.exceptions import AgentDataGenException
from .task_designer import TaskDesigner


class TaskGenerator(BaseModule):
    """Main class for task generation module"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize task generation module
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        self.task_designer = None
        
        self.tasks_per_difficulty = 3  # Number of tasks generated per difficulty level
        self.max_workers = 32  # Number of concurrent workers
        
    def _setup(self):
        """Setup module components"""
        # Update configuration
        config = self.config or {}
        self.tasks_per_difficulty = config.get('tasks_per_difficulty', 3)
        self.max_workers = config.get('max_workers', 32)
        
        self.task_designer = TaskDesigner(self.config, self.logger)
        self.task_designer.initialize()
        
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process task generation
        
        Args:
            input_data: Input data containing agent and tool information
            **kwargs: Other parameters
            
        Returns:
            Generated task data
        """
        try:
            agents = input_data.get('agents', [])
            tools_data = input_data.get('tools_data', {})
            
            if not agents:
                raise ValueError("No agents provided for task generation")
            
            if not tools_data:
                raise ValueError("No tools data provided for task generation")
            
            # Calculate total number of tasks
            total_expected_tasks = len(agents) * len(DifficultyLevel) * self.tasks_per_difficulty
            self.logger.info(f"Starting task generation for {len(agents)} agents")
            self.logger.info(f"Expected total tasks: {total_expected_tasks}")
            
            task_params = self._generate_task_parameters(agents, tools_data)
            all_tasks = self._generate_tasks_concurrently(task_params)
            
            if all_tasks:
                self.task_designer.save_batch_tasks(all_tasks)
            
            self.logger.info(f"Successfully generated {len(all_tasks)} tasks for {len(agents)} agents")
            
            return {
                'tasks': all_tasks,
                'total_tasks': len(all_tasks),
                'total_agents': len(agents),
                'generation_summary': {
                    'tasks_per_agent': len(all_tasks) / len(agents) if agents else 0,
                    'difficulty_distribution': self._calculate_difficulty_distribution(all_tasks),
                    'success_rate': len(all_tasks) / total_expected_tasks if total_expected_tasks > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Task generation failed: {e}")
            raise AgentDataGenException(f"Failed to generate tasks: {e}")
    
    def _generate_task_parameters(self, agents: List[Dict[str, Any]], 
                                 tools_data: Dict[str, Any]) -> List[Tuple[str, List[str], DifficultyLevel, int]]:
        """
        Generate all task parameter combinations
        
        Args:
            agents: List of agents
            tools_data: Tool data
            
        Returns:
            Task parameter list: [(agent_id, agent_tools, difficulty, task_index), ...]
        """
        task_params = []
        
        for agent in agents:
            if isinstance(agent, dict):
                agent_id = agent.get('id')
                agent_tools = agent.get('tools', [])
            else:
                agent_id = agent.id
                agent_tools = agent.tools
            
            # Get detailed tool information for the agent
            tools_info = self.task_designer._get_tools_info(agent_tools, tools_data)
            
            # Skip this agent if no valid tools found
            if not tools_info:
                self.logger.warning(f"No valid tools found for agent {agent_id}, skipping")
                continue
            
            # Generate multiple task parameters for each difficulty level
            for difficulty in DifficultyLevel:
                for task_index in range(self.tasks_per_difficulty):
                    task_params.append((agent_id, tools_info, difficulty, task_index))
        
        return task_params
    
    def _generate_tasks_concurrently(self, task_params: List[Tuple[str, List[Dict[str, Any]], DifficultyLevel, int]]) -> List[Task]:
        """
        Generate all tasks concurrently
        
        Args:
            task_params: Task parameter list
            
        Returns:
            List of generated tasks
        """
        all_tasks = []
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_params = {}
            for params in task_params:
                agent_id, tools_info, difficulty, task_index = params
                future = executor.submit(
                    self.task_designer.generate_single_task,
                    agent_id=agent_id,
                    tools_info=tools_info,
                    difficulty=difficulty
                )
                future_to_params[future] = params
            
            # Collect results
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                agent_id, tools_info, difficulty, task_index = params
                
                try:
                    task = future.result()
                    if task:
                        all_tasks.append(task)
                        if len(all_tasks) % 50 == 0:  # Output progress every 50 tasks
                            self.logger.info(f"Generated {len(all_tasks)} tasks so far...")
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Failed to generate task for agent {agent_id}, difficulty {difficulty.value}: {e}")
        
        if failed_count > 0:
            self.logger.warning(f"Failed to generate {failed_count} tasks out of {len(task_params)} total")
        
        return all_tasks
    
    def _calculate_difficulty_distribution(self, tasks: List[Task]) -> Dict[str, int]:
        """Calculate task difficulty distribution"""
        distribution = {'simple': 0, 'medium': 0, 'complex': 0}
        
        for task in tasks:
            difficulty = task.difficulty.value if hasattr(task.difficulty, 'value') else task.difficulty
            if difficulty in distribution:
                distribution[difficulty] += 1
        
        return distribution
