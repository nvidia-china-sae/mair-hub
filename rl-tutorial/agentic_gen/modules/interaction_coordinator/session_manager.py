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
Unified Session Manager
Manages unified conversation sessions between user, agent and tool executor
"""

import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from core.base_module import BaseModule
from core.models import Task, AgentConfig, UserPersona, ConversationTurn, InteractionSession, Trajectory
from core.exceptions import AgentDataGenException
from utils.data_processor import DataProcessor
from utils.file_manager import FileManager


class SessionManager(BaseModule):
    """Unified Session Manager"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize unified session manager
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.data_processor = None
        self.file_manager = None
        
        # Current session state
        self.current_session = None
        self.inference_data = ""
        
        # Configuration
        self.max_turns = 20
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize file manager
        data_path = settings.get_data_path('trajectories')
        self.file_manager = FileManager(data_path, self.logger)
        
        # Update configuration
        config = self.config or {}
        self.max_turns = config.get('max_turns', 20)
    
    def create_session(self, task: Task, agent_config: AgentConfig, user_persona: UserPersona) -> InteractionSession:
        """
        Create unified session
        
        Args:
            task: Task object
            agent_config: Agent configuration
            user_persona: User persona
            
        Returns:
            Interaction session object
        """
        try:
            # Generate session ID
            session_id = self.data_processor.generate_id('session', {
                'task_id': task.id,
                'agent_id': agent_config.id,
            })
            
            # Create session object
            session = InteractionSession(
                id=session_id,
                task_id=task.id,
                agent_id=agent_config.id,
                session_state={
                    'persona_info': {
                        'personality_type': user_persona.personality_type.value,
                        'style_type': user_persona.style_type.value,
                        'characteristics': user_persona.metadata
                    },
                    'agent_tools': agent_config.tools,
                    'turn_count': 0
                }
            )
            
            # Set current session
            self.current_session = session
            self.inference_data = ""
            
            self.logger.info(f"Created unified session: {session_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to create unified session: {e}")
            raise AgentDataGenException(f"Session creation failed: {e}")
    
    def add_message(self, sender: str, recipient: str, message: str):
        """
        Add message to conversation history
        
        Args:
            sender: Sender ("user", "agent", "execution")
            recipient: Recipient ("user", "agent", "execution")
            message: Message content
        """
        try:
            if not self.current_session:
                raise AgentDataGenException("No active session to add message")
            
            # Create ConversationTurn object
            turn = ConversationTurn(
                speaker=sender,
                recipient=recipient,
                message=message,
                metadata={
                    'turn_index': len(self.current_session.turns)
                }
            )
            
            # Add to session
            self.current_session.turns.append(turn)
            self.current_session.session_state['turn_count'] = len(self.current_session.turns)
            
            self.logger.debug(f"Added message: {sender} -> {recipient}")
            
        except Exception as e:
            self.logger.error(f"Failed to add message: {e}")
    
    def get_last_recipient(self) -> str:
        """Get recipient of last message"""
        if not self.current_session or not self.current_session.turns:
            return "user"
        return self.current_session.turns[-1].recipient
    
    def get_last_message(self) -> Dict[str, Any]:
        """Get last message"""
        if not self.current_session or not self.current_session.turns:
            return {}
        last_turn = self.current_session.turns[-1]
        return {
            "sender": last_turn.speaker,
            "recipient": last_turn.recipient,
            "message": last_turn.message
        }
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        try:
            if not self.current_session:
                return ""
                
            history_lines = []
            
            for turn in self.current_session.turns:
                sender = turn.speaker
                message = turn.message
                
                if sender == "user":
                    history_lines.append(f"user: {message}")
                elif sender == "agent":
                    history_lines.append(f"agent: {message}")
                elif sender == "execution":
                    # Handle display of execution results
                    if isinstance(message, list):
                        # If execution result list, format display
                        for result in message:
                            history_lines.append(f"execution: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    else:
                        history_lines.append(f"execution: {message}")
            
            return "\n".join(history_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return ""
    
    def should_end_conversation(self) -> bool:
        """Determine if conversation should end"""
        try:
            if not self.current_session:
                return True
            
            turn_count = len(self.current_session.turns)
            
            # Check maximum turns
            if turn_count >= self.max_turns:
                return True
            
            # Check for "finish conversation" message
            if self.current_session.turns:
                last_message = self.current_session.turns[-1].message
                if "finish conversation" in str(last_message).lower():
                    return True
            
            return False
            
        except Exception:
            return True
    
    def finalize_session(self) -> Trajectory:
        """Complete session and generate trajectory"""
        try:
            if not self.current_session:
                raise AgentDataGenException("No active session to finalize")
            
            # Update session status
            self.current_session.status = "completed"
            self.current_session.ended_at = datetime.now()
            
            # Generate trajectory ID
            trajectory_id = self.data_processor.generate_id('trajectory', {
                'session_id': self.current_session.id,
                'task_id': self.current_session.task_id,
                'agent_id': self.current_session.agent_id
            })
            
            # Create trajectory object
            trajectory = Trajectory(
                id=trajectory_id,
                session=self.current_session
            )
            
            self.logger.info(f"Finalized session: {self.current_session.id}")
            return trajectory
            
        except Exception as e:
            self.logger.error(f"Failed to finalize session: {e}")
            raise AgentDataGenException(f"Session finalization failed: {e}")
    
    def save_session(self, trajectory: Trajectory) -> str:
        """Save session trajectory"""
        try:
            # Convert to training data format
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
