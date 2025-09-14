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
Task Designer
Responsible for designing multi-turn conversation tasks and corresponding scoring criteria for agents
"""

import json
import random
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from core.base_module import BaseModule
from core.models import Task, TaskRubric, DifficultyLevel, TaskType
from core.exceptions import AgentDataGenException
from utils.llm_client import LLMClient
from utils.data_processor import DataProcessor
from utils.file_manager import FileManager


class TaskDesigner(BaseModule):
    """Task Designer"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize the task designer
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.llm_client = None
        self.data_processor = None
        self.file_manager = None
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        llm_config = settings.get_llm_config()
        llm_config['provider'] = settings.DEFAULT_LLM_PROVIDER
        self.llm_client = LLMClient(llm_config, self.logger)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize file manager
        data_path = settings.get_data_path('tasks')
        self.file_manager = FileManager(data_path, self.logger)
    
    def generate_single_task(self, agent_id: str, tools_info: List[Dict[str, Any]], 
                            difficulty: DifficultyLevel) -> Optional[Task]:
        """Generate a single task"""
        try:
            from config.prompts.task_prompts import TaskPrompts
            
            # Prepare tool information
            tools_details = self._format_tools_for_prompt(tools_info)
            available_tools = [tool['name'] for tool in tools_info]
            
            prompts = TaskPrompts()
            prompt = prompts.TASK_GENERATION.format(
                available_tools=available_tools,
                tools_details=tools_details,
                difficulty=difficulty.value
            )
            
            response = self.llm_client.generate_completion(prompt=prompt)
            task_data = self.llm_client.parse_json_response(response)
            
            if not task_data:
                self.logger.error("Failed to parse task generation response")
                return None
            
            if not self._validate_task_data(task_data, available_tools):
                self.logger.warning("Generated task failed validation")
                return None
            
            task = self._create_task_from_data(agent_id, task_data, difficulty)
            
            return task
            
        except Exception as e:
            self.logger.error(f"Failed to generate single task: {e}")
            return None
    
    def _get_tools_info(self, tool_ids: List[str], tools_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed tool information"""
        tools_info = []
        
        for tool_id in tool_ids:
            if tool_id in tools_data:
                tools_info.append(tools_data[tool_id])
            else:
                self.logger.warning(f"Tool {tool_id} not found in tools_data")
        
        return tools_info
    
    def _format_tools_for_prompt(self, tools_info: List[Dict[str, Any]]) -> str:
        """Format tool information for prompt"""
        tool_descriptions = []
        
        for tool in tools_info:
            tool_name = tool.get('name', '')
            tool_desc = tool.get('description', '')
            parameters = tool.get('parameters', [])
            
            # Build parameter information
            param_list = []
            for param in parameters:
                param_name = param.get('name', '')
                param_type = param.get('type', '')
                param_desc = param.get('description', '')
                required = param.get('required', False)
                
                param_info = f"  - {param_name} ({param_type})"
                if required:
                    param_info += " [Required]"
                param_info += f": {param_desc}"
                param_list.append(param_info)
            
            # Build tool description
            tool_text = f"**{tool_name}**\\nFunction: {tool_desc}"
            if param_list:
                tool_text += "\\nParameters:\\n" + "\\n".join(param_list)
            
            tool_descriptions.append(tool_text)
        
        return "\\n\\n".join(tool_descriptions)
    
    def _validate_task_data(self, task_data: Dict[str, Any], available_tools: List[str]) -> bool:
        """Validate generated task data"""
        try:
            # Check basic structure
            if 'task' not in task_data or 'rubric' not in task_data:
                return False
            
            task_info = task_data['task']
            rubric_info = task_data['rubric']
            
            # Check basic task fields
            required_task_fields = ['title', 'description', 'difficulty']
            for field in required_task_fields:
                if field not in task_info:
                    return False
            
            # Check rubric fields
            required_rubric_fields = ['checkpoints', 'success_criteria']
            for field in required_rubric_fields:
                if field not in rubric_info:
                    return False
            
            # Check if checkpoints use available tools
            checkpoints = rubric_info.get('checkpoints', [])
            if not checkpoints:
                return False
            
            # Extract tool names from checkpoints and validate
            used_tools = []
            for checkpoint in checkpoints:
                if '(' in checkpoint:
                    tool_name = checkpoint.split('(')[0].strip()
                    used_tools.append(tool_name)
            
            for tool in used_tools:
                if tool not in available_tools:
                    self.logger.warning(f"Tool {tool} in checkpoints not available in agent tools")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Task validation error: {e}")
            return False
    
    def _create_task_from_data(self, agent_id: str, task_data: Dict[str, Any], 
                             difficulty: DifficultyLevel) -> Task:
        """Create Task object from data"""
        task_info = task_data['task']
        rubric_info = task_data['rubric']
        
        task_id = self.data_processor.generate_id('task', {
            'agent_id': agent_id,
            'title': task_info.get('title', ''),
            'difficulty': difficulty.value
        })
        
        checkpoints = rubric_info.get('checkpoints', [])
        expected_tools = []
        for checkpoint in checkpoints:
            if '(' in checkpoint:
                tool_name = checkpoint.split('(')[0].strip()
                expected_tools.append(tool_name)
        
        # Create TaskRubric
        rubric = TaskRubric(
            success_criteria=rubric_info.get('success_criteria', []),
            tool_usage_expectations=rubric_info.get('tool_usage_expectations', []),
            checkpoints=checkpoints
        )
        
        task = Task(
            id=task_id,
            agent_id=agent_id,
            title=task_info.get('title', ''),
            description=task_info.get('description', ''),
            difficulty=difficulty,
            task_type=TaskType.MULTI_TURN,
            expected_tools=expected_tools,
            rubric=rubric,
            metadata={
                'expected_turns': task_info.get('expected_turns', '4-8'),
                'generated_at': datetime.now().isoformat()
            }
        )
        
        return task
    

    def save_batch_tasks(self, tasks: List[Task]):
        """Batch save tasks"""
        try:
            # Convert to serializable format
            tasks_data = []
            for task in tasks:
                task_dict = {
                    'id': task.id,
                    'agent_id': task.agent_id,
                    'title': task.title,
                    'description': task.description,
                    'difficulty': task.difficulty.value,
                    'task_type': task.task_type.value,
                    'expected_tools': task.expected_tools,
                    'rubric': {
                        'success_criteria': task.rubric.success_criteria,
                        'tool_usage_expectations': task.rubric.tool_usage_expectations,
                        'checkpoints': task.rubric.checkpoints
                    },
                    'metadata': task.metadata,
                    'created_at': task.created_at.isoformat()
                }
                tasks_data.append(task_dict)
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tasks_batch_{timestamp}.json"
            
            self.file_manager.save_json(tasks_data, filename)
            self.logger.info(f"Saved {len(tasks)} tasks to {filename}")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save batch tasks: {e}")
            raise AgentDataGenException(f"Failed to save batch tasks: {e}")
