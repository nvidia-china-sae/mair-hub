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
User Simulator
Responsible for simulating real users in multi-turn conversational interactions with agents
"""

import json
from typing import Dict, Any
import logging

from core.base_module import BaseModule
from core.models import Task, UserPersona
from core.exceptions import AgentDataGenException
from utils.llm_client import LLMClient
from config.prompts.user_prompts import UserPrompts
from .user_persona_generator import UserPersonaGenerator


class UserSimulator(BaseModule):
    """User Simulator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize user simulator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.llm_client = None
        self.persona_generator = None
        self.prompts = UserPrompts()
        
        # Current state
        self.current_persona = None
        self.current_task = None
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize LLM client
        llm_config = settings.get_llm_config()
        llm_config['provider'] = settings.DEFAULT_LLM_PROVIDER
        self.llm_client = LLMClient(llm_config, self.logger)
        
        # Initialize persona generator
        self.persona_generator = UserPersonaGenerator(logger=self.logger)
        self.persona_generator.initialize()
    
    def initialize_for_task(self, task: Task, user_persona: UserPersona):
        """
        Initialize user simulator for task
        
        Args:
            task: Task object
            user_persona: User persona
        """
        self.current_task = task
        self.current_persona = user_persona
        self.logger.info(f"Initialized user simulator for task {task.id} with persona {user_persona.id}")
    
    def generate_initial_message(self) -> str:
        """Generate initial user message"""
        try:
            if not self.current_task or not self.current_persona:
                raise AgentDataGenException("No active task or persona for message generation")
            
            user_characteristics = self.prompts.USER_CHARACTERISTICS_TEMPLATE.format(
                personality_description=self.current_persona.metadata.get('personality_description', ''),
                style_description=self.current_persona.metadata.get('style_description', '')
            )
            
            system_prompt = self.prompts.USER_SIMULATION_SYSTEM.format(
                user_characteristics=user_characteristics,
                task_instruction=self.current_task.description
            )
            
            response = self.llm_client.generate_completion(
                prompt=self.prompts.INIT_CONVERSATION,
                system_prompt=system_prompt
            )
            
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate initial message: {e}")
            return "Hello, I need some help."  # Fallback to default message
    
    def respond_to_agent(self, agent_message: str, conversation_history: str = "") -> str:
        """
        Respond to agent's message
        
        Args:
            agent_message: Message sent by agent
            conversation_history: Conversation history
            
        Returns:
            User's response message
        """
        try:
            if not self.current_persona or not self.current_task:
                raise AgentDataGenException("No active persona or task for response generation")
            
            user_characteristics = self.prompts.USER_CHARACTERISTICS_TEMPLATE.format(
                personality_description=self.current_persona.metadata.get('personality_description', ''),
                style_description=self.current_persona.metadata.get('style_description', '')
            )
            
            system_prompt = self.prompts.USER_SIMULATION_SYSTEM.format(
                user_characteristics=user_characteristics,
                task_instruction=self.current_task.description
            )
            
            user_prompt = self.prompts.USER_RESPONSE_PROMPT.format(
                conversation_history=conversation_history
            )
            
            response = self.llm_client.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            user_response = response.content.strip()
            
            return user_response
            
        except Exception as e:
            self.logger.error(f"Failed to generate user response: {e}")
            return "Okay, I understand."