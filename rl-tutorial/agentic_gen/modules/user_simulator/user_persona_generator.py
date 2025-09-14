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
User Persona Generator
Responsible for generating diverse user persona configurations
"""

import random
from typing import Dict, Any, List
import logging
from datetime import datetime

from core.base_module import BaseModule
from core.models import UserPersona, UserPersonalityType, InteractionStyle
from core.exceptions import AgentDataGenException
from utils.data_processor import DataProcessor
from utils.file_manager import FileManager
from config.prompts.user_prompts import UserPrompts


class UserPersonaGenerator(BaseModule):
    """User Persona Generator"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize user persona generator
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.data_processor = None
        self.file_manager = None
        self.prompts = UserPrompts()
        
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize file manager
        data_path = settings.get_data_path('user_personas')
        self.file_manager = FileManager(data_path, self.logger)
    
    def process(self) -> UserPersona:
        """
        Generate user persona
        
        Args:
            input_data: Input data
            **kwargs: Other parameters
            
        Returns:
            Generated user persona
        """
        
        return self._generate_single_persona()
    
    def _generate_single_persona(self) -> UserPersona:
        """Generate single user persona"""
        try:
            # Randomly select personality type and interaction style
            personality_type = random.choice(list(UserPersonalityType))
            style_type = random.choice(list(InteractionStyle))
            
            # Generate persona ID
            persona_id = self.data_processor.generate_id('user_persona', {
                'personality': personality_type.value,
                'style': style_type.value,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate persona name
            name = f"{personality_type.value}_{style_type.value}_user"
            
            # Create UserPersona object
            persona = UserPersona(
                id=persona_id,
                name=name,
                personality_type=personality_type,
                style_type=style_type,
                metadata={
                    'personality_description': self.prompts.PERSONALITY_DESCRIPTIONS[personality_type.value],
                    'style_description': self.prompts.STYLE_DESCRIPTIONS[style_type.value],
                    'generated_at': datetime.now().isoformat()
                }
            )
            
            return persona
            
        except Exception as e:
            self.logger.error(f"Failed to generate single persona: {e}")
            return None
