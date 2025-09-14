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
Custom exception class definitions
Defines all exception types used in the system
"""


class AgentDataGenException(Exception):
    """Base exception class"""
    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(AgentDataGenException):
    """Configuration error"""
    pass


class ModelValidationError(AgentDataGenException):
    """Data model validation error"""
    pass


class LLMApiError(AgentDataGenException):
    """LLM API call error"""
    pass


class ToolExecutionError(AgentDataGenException):
    """Tool execution error"""
    pass


class ScenarioGenerationError(AgentDataGenException):
    """Scenario generation error"""
    pass


class ToolDesignError(AgentDataGenException):
    """Tool design error"""
    pass


class AgentSynthesisError(AgentDataGenException):
    """Agent synthesis error"""
    pass


class TaskGenerationError(AgentDataGenException):
    """Task generation error"""
    pass


class UserSimulationError(AgentDataGenException):
    """User simulation error"""
    pass


class TrajectoryGenerationError(AgentDataGenException):
    """Trajectory generation error"""
    pass


class QualityEvaluationError(AgentDataGenException):
    """Quality evaluation error"""
    pass


class DataStorageError(AgentDataGenException):
    """Data storage error"""
    pass


class PipelineExecutionError(AgentDataGenException):
    """Pipeline execution error"""
    pass


class RegistryError(AgentDataGenException):
    """Registry error"""
    pass


class ValidationError(AgentDataGenException):
    """Validation error"""
    pass 