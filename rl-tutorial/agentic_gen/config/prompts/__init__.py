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
提示词模板模块
管理各个功能模块的提示词模板
"""

from .scenario_prompts import ScenarioPrompts
from .tool_prompts import ToolPrompts
from .agent_prompts import AgentPrompts
from .task_prompts import TaskPrompts
# from .evaluation_prompts import EvaluationPrompts

__all__ = [
    'ScenarioPrompts',
    'ToolPrompts',
    'AgentPrompts', 
    'TaskPrompts',
    # 'EvaluationPrompts'
]