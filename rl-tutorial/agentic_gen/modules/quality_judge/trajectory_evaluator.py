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
Trajectory Evaluator
Evaluates the quality of multi-turn agent interaction trajectories
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from core.base_module import BaseModule
from core.models import Trajectory, TrajectoryScore, Task, ConversationTurn
from core.exceptions import QualityEvaluationError
from config.prompts.evaluation_prompts import EvaluationPrompts
from utils.llm_client import LLMClient, LLMResponse
from utils.logger import setup_logger
from utils.file_manager import FileManager


class TrajectoryEvaluator(BaseModule):
    """
    Trajectory Evaluator
    Evaluates the quality of multi-turn agent interaction trajectories
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize trajectory evaluator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        self.llm_client = None
        self.file_manager = None
        self.evaluation_prompts = EvaluationPrompts()
        
    def _setup(self):
        """Setup module"""
        from config.settings import settings
        
        llm_config = settings.get_llm_config()
        llm_config['provider'] = settings.DEFAULT_LLM_PROVIDER
        self.llm_client = LLMClient(llm_config, self.logger)
        data_path = settings.get_data_path('trajectory_evaluations')
        self.file_manager = FileManager(data_path, self.logger)
        
        self.logger.info("TrajectoryEvaluator initialized successfully")
    
    def evaluate_trajectory(
        self, 
        trajectory: Trajectory, 
        task: Optional[Task] = None,
        **kwargs
    ) -> TrajectoryScore:
        """
        Evaluate the quality of a single trajectory
        
        Args:
            trajectory: Trajectory to evaluate
            task: Corresponding task information (optional)
            **kwargs: Additional parameters
            
        Returns:
            Trajectory scoring result
        """
        try:
            self.logger.info(f"Evaluating trajectory: {trajectory.id}")
            
            evaluation_data = self._prepare_evaluation_data(trajectory, task)
            evaluation_prompt = self._generate_evaluation_prompt(evaluation_data)
            
            evaluation_response = self.llm_client.generate_completion(
                prompt=evaluation_prompt,
                system_prompt=self.evaluation_prompts.TRAJECTORY_EVALUATION_SYSTEM,
                temperature=0.1,
            )
            
            evaluation_result = self._parse_evaluation_result(evaluation_response)
            
            trajectory.evaluation_score = TrajectoryScore(
                overall_score=evaluation_result.get("overall_score", 0)
            )

            self.save_trajectory_evaluation(trajectory)
            return trajectory
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate trajectory {trajectory.id}: {e}")
            raise QualityEvaluationError(f"Trajectory evaluation failed: {e}")
    
    
    def _prepare_evaluation_data(
        self, 
        trajectory: Trajectory, 
        task: Optional[Task] = None
    ) -> Dict[str, Any]:
        """
        Prepare evaluation data
        
        Args:
            trajectory: Trajectory object
            task: Task object
            
        Returns:
            Evaluation data dictionary
        """
        # Extract conversation history
        conversation_history = []
        tool_results = []
        
        for turn in trajectory.session.turns:
            turn_data = {
                "speaker": turn.speaker,
                "recipient": turn.recipient,
                "message": turn.message,
            }
            conversation_history.append(turn_data)
            
            # Extract tool execution results
            if turn.speaker == "execution":          
                tool_results.append({
                    "result": turn.message
                })

        
        # Prepare task information
        task_info = {}
        if task:
            task_info = {
                "description": task.description,
                "expected_tools": task.expected_tools,
                "success_criteria": task.rubric.success_criteria if task.rubric else []
            }
        
        return {
            "trajectory_id": trajectory.id,
            "task_info": task_info,
            "conversation_history": conversation_history,
            "tool_results": tool_results,
        }
    
    def _generate_evaluation_prompt(self, evaluation_data: Dict[str, Any]) -> str:
        """
        Generate evaluation prompt
        
        Args:
            evaluation_data: Evaluation data
            
        Returns:
            Evaluation prompt
        """
        task_info = evaluation_data["task_info"]
        
        return self.evaluation_prompts.TRAJECTORY_EVALUATION_USER.format(
            task_description=task_info.get("description", ""),
            tool_usage_expectations="\n".join([f"- {expection}" for expection in task_info.get("tool_usage_expectations", [])]),
            conversation_history=json.dumps(evaluation_data["conversation_history"], indent=2, ensure_ascii=False),
            tool_results=json.dumps(evaluation_data["tool_results"], indent=2, ensure_ascii=False)
        )
    
    def _parse_evaluation_result(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Parse evaluation results
        
        Args:
            response: LLM response
            
        Returns:
            Parsed evaluation results
        """
        try:
            evaluation_result = self.llm_client.parse_json_response(response)
            
            if "overall_score" not in evaluation_result:
                raise QualityEvaluationError(f"Missing required field in evaluation result: overall_score")
            
            # Validate score range
            overall_score = evaluation_result.get("overall_score", 0)
            if not 0 <= overall_score <= 5:
                raise QualityEvaluationError(f"Invalid overall score: {overall_score}")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Failed to parse evaluation result: {e}")
            raise QualityEvaluationError(f"Failed to parse evaluation result: {e}")

    def prefilter_trajectory(self, trajectory: Trajectory) -> bool:
        """
        Pre-filter trajectory
        
        Args:
            trajectory: Trajectory object
            
        Returns:
            Whether it passes pre-filter checks
        """
        # Basic structure check
        if not trajectory.session or not trajectory.session.turns:
            self.logger.debug(f"Trajectory {trajectory.id}: No session or turns")
            return False

        # Check for at least one user-agent interaction
        user_turns = [turn for turn in trajectory.session.turns if turn.speaker == "user"]
        agent_turns = [turn for turn in trajectory.session.turns if turn.speaker == "agent"]
        
        if not user_turns or not agent_turns:
            return False

        # Get last conversation turn
        last_turn = trajectory.session.turns[-1]
        
        # Check if last message was sent by user
        if last_turn.speaker != "user":
            return False
        
        # Check last message content
        last_message = last_turn.message
        
        message_text = last_message.lower()
        
        
        # Check if contains "finish conversation"
        if "finish conversation" not in message_text:
            return False
        
        self.logger.debug(f"Trajectory {trajectory.id}: Passed prefilter checks")
        return True
    
    def save_trajectory_evaluation(
        self, 
        trajectory: Trajectory, 
    ) -> Dict[str, Any]:
        """
        Save evaluated trajectory data
        
        Args:
            trajectory: Evaluated trajectory object
            
        Returns:
            Save result information dictionary
        """
        try:
            training_data = trajectory.to_training_format()
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{trajectory.id}_{timestamp}.json"
            
            self.file_manager.save_json(training_data, filename)
            self.logger.info(f"Saved trajectory to {filename}")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            raise AgentDataGenException(f"Session save failed: {e}")
