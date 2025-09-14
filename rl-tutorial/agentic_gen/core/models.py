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
Core data model definitions
Defines all data structures used in the system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
from datetime import datetime


class DifficultyLevel(Enum):
    """Task difficulty level"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class TaskType(Enum):
    """Task type"""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"


class UserPersonalityType(Enum):
    """User personality type"""
    FRIENDLY = "friendly"
    IMPATIENT = "impatient"
    CAUTIOUS = "cautious"
    CASUAL = "casual"


class InteractionStyle(Enum):
    """Interaction style"""
    FORMAL = "formal"
    INFORMAL = "informal"
    LIFE_ORIENTED = "life_oriented"


@dataclass
class Scenario:
    """Scenario data model"""
    id: str
    name: str
    description: str
    domain: str
    category: str
    context: str
    use_cases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolParameter:
    """Tool parameter"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """Tool data model"""
    id: str
    name: str
    description: str
    category: str
    scenario_ids: List[str] = field(default_factory=list)
    parameters: List[ToolParameter] = field(default_factory=list)
    return_type: str = "dict"
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_function_schema(self) -> Dict[str, Any]:
        """Convert to function call schema format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


@dataclass
class AgentConfig:
    """Agent configuration"""
    id: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)  # tool_ids
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskRubric:
    """Task scoring criteria"""
    success_criteria: List[str] = field(default_factory=list)
    tool_usage_expectations: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Task data model"""
    id: str
    agent_id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    task_type: TaskType
    expected_tools: List[str] = field(default_factory=list)
    rubric: TaskRubric = field(default_factory=TaskRubric)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserPersona:
    """User persona model"""
    id: str
    name: str
    personality_type: UserPersonalityType
    style_type: InteractionStyle
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationTurn:
    """Conversation turn"""
    speaker: str  # "user", "agent", "execution"
    recipient: str  # "user", "agent", "execution"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionSession:
    """Interaction session"""
    id: str
    task_id: str
    agent_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    session_state: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, completed, failed
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryScore:
    """Trajectory score"""
    overall_score: float
    pass_threshold: float = 4.0
    
    @property
    def passed(self) -> bool:
        return self.overall_score >= self.pass_threshold


@dataclass
class Trajectory:
    """Complete interaction trajectory"""
    id: str
    session: InteractionSession
    evaluation_score: Optional[TrajectoryScore] = None
    quality_tags: List[str] = field(default_factory=list)
    is_high_quality: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to training data format"""
        messages = []
        for turn in self.session.turns:
            if turn.speaker == "user":
                messages.append({
                    "role": "user", 
                    "content": turn.message,
                    "recipient": turn.recipient
                })
            elif turn.speaker == "agent":
                messages.append({
                    "role": "assistant", 
                    "content": turn.message,
                    "recipient": turn.recipient
                })
            elif turn.speaker == "execution":
                messages.append({
                    "role": "execution",
                    "content": turn.message,
                    "recipient": turn.recipient
                })
        
        return {
            "trajectory_id": self.id,
            "task_id": self.session.task_id,
            "agent_id": self.session.agent_id,
            "messages": messages,
            "score": self.evaluation_score.overall_score if self.evaluation_score else None,
            "metadata": {
                "quality_tags": self.quality_tags,
                "is_high_quality": self.is_high_quality,
                "session_metadata": self.session.metadata
            }
        }


# Utility functions
def serialize_dataclass(obj) -> str:
    """Serialize dataclass object to JSON string"""
    if hasattr(obj, '__dataclass_fields__'):
        # Process dataclass
        data = {}
        for field_name, field_def in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            if isinstance(value, datetime):
                data[field_name] = value.isoformat()
            elif isinstance(value, Enum):
                data[field_name] = value.value
            elif hasattr(value, '__dataclass_fields__'):
                data[field_name] = serialize_dataclass(value)
            elif isinstance(value, list):
                data[field_name] = [serialize_dataclass(item) if hasattr(item, '__dataclass_fields__') else item for item in value]
            else:
                data[field_name] = value
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def deserialize_dataclass(cls, data: Union[str, dict]):
    """Deserialize dataclass object from JSON string or dictionary"""
    if isinstance(data, str):
        data = json.loads(data)
    
    # Deserialization logic needs to be implemented based on specific class
    # Simplified implementation, actual projects may need more complex handling
    return cls(**data) 