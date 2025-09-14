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
Interaction Coordinator
Responsible for coordinating the entire multi-agent data generation process
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from core.base_module import BaseModule
from core.models import Task, AgentConfig, Trajectory
from core.exceptions import AgentDataGenException
from utils.data_processor import DataProcessor
from .session_manager import SessionManager
from modules.user_simulator import UserSimulator
from modules.agent_simulator import AgentSimulator
from modules.tool_execution import ToolExecutionSimulator


class InteractionCoordinator(BaseModule):
    """Interaction Coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize interaction coordinator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.data_processor = None
        
        # Components
        self.session_manager = None
        self.user_simulator = None
        self.agent_simulator = None
        self.tool_execution_simulator = None
        
        # Configuration
        self.max_turns = 10
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize sub-modules
        self.session_manager = SessionManager(logger=self.logger)
        self.session_manager.initialize()
        
        self.user_simulator = UserSimulator(logger=self.logger)
        self.user_simulator.initialize()
        
        self.agent_simulator = AgentSimulator(logger=self.logger)
        self.agent_simulator.initialize()
        
        self.tool_execution_simulator = ToolExecutionSimulator(logger=self.logger)
        self.tool_execution_simulator.initialize()
        
        # Update configuration
        config = self.config or {}
        self.max_turns = config.get('max_turns', 20)
        print(f"max_turns: {self.max_turns}")
            
    def execute_single_interaction(self, task: Task, agent_config: AgentConfig, tools_info: Dict[str, Any]) -> Trajectory:
        """
        Execute single interaction session
        
        Args:
            task: Task object
            agent_config: Agent configuration
            tools_info: Tools information
            
        Returns:
            Interaction trajectory
        """
        try:
            self.logger.info(f"Executing interaction: Task {task.id} with Agent {agent_config.id}")
            
            user_persona = self.user_simulator.persona_generator.process()
            session = self.session_manager.create_session(task, agent_config, user_persona)
            
            self.user_simulator.initialize_for_task(task, user_persona)
            self.agent_simulator.initialize_for_agent(agent_config, tools_info)
            self.tool_execution_simulator.initialize_tools(tools_info)
            
            init_message = self.user_simulator.generate_initial_message()
            self.session_manager.add_message("user", "agent", init_message)
            
            self._execute_conversation_loop()
            
            trajectory = self.session_manager.finalize_session()
            
            self.session_manager.save_session(trajectory)
            
            self.logger.info(f"Interaction completed: {trajectory.id}")
            return trajectory
            
        except Exception as e:
            self.logger.error(f"Single interaction failed: {e}")
            raise AgentDataGenException(f"Interaction execution failed: {e}")
    
    def _execute_conversation_loop(self):
        """
        Execute conversation loop
        """
        try:
            turn_count = 0
            
            while turn_count < self.max_turns:
                turn_count += 1
                
                last_recipient = self.session_manager.get_last_recipient()
                if last_recipient == "user":
                    last_message = self.session_manager.get_last_message()
                    conversation_history = self.session_manager.get_conversation_history()
                    
                    user_response = self.user_simulator.respond_to_agent(
                        last_message.get("message", ""),
                        conversation_history
                    )                    
                    if "finish conversation" in user_response.lower():
                        self.session_manager.add_message("user", "agent", user_response)
                        self.logger.info("User indicated conversation completion")
                        break
                    
                    self.session_manager.add_message("user", "agent", user_response)
                    
                elif last_recipient == "agent":
                    history_messages = self.session_manager.get_conversation_history()
                    current_message = self.agent_simulator.respond(history_messages)
                    
                    self.session_manager.add_message(
                        current_message["sender"],
                        current_message["recipient"],
                        current_message["message"]
                    )
                    
                elif last_recipient == "execution":
                    last_message = self.session_manager.get_last_message()
                    execution_results = self.tool_execution_simulator.execute_agent_message(
                        last_message.get("message", "")
                    )
                    self.session_manager.add_message("execution", "agent", execution_results)
                
                if self.session_manager.should_end_conversation():
                    break
            
            self.logger.info(f"Conversation completed after {turn_count} turns")
            
        except Exception as e:
            self.logger.error(f"Conversation loop failed: {e}")
            raise AgentDataGenException(f"Conversation execution failed: {e}")
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        stats = {
            'max_turns': self.max_turns,
            'module_name': self.__class__.__name__,
            'initialized': self._initialized,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.tool_execution_simulator:
            execution_stats = self.tool_execution_simulator.get_execution_stats()
            stats['execution_stats'] = execution_stats
        
        if self.session_manager:
            session_summary = self.session_manager.get_session_summary()
            stats['session_summary'] = session_summary
        
        return stats
