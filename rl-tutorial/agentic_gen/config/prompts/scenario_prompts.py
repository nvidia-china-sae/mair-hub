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
Prompt templates for scenario generation
"""


class ScenarioPrompts:
    """Scenario generation prompt templates"""
    
    SCENARIO_GENERATION = """
You are a professional application scenario designer responsible for generating rich and diverse application scenarios for a multi-agent data synthesis project.

Please generate {count} specific application scenarios based on the following domain:  
Domain: {domain}

Each scenario should include:  
1. Scenario name (concise and clear)  
2. Detailed description (100-200 words)  
3. Application   
4. Typical user needs

Requirements:  
- Scenarios should be realistic, practical and high frequency
- Descriptions must be specific and avoid being too abstract  
- Use cases should cover different situations  
- Ensure sufficient differences between scenarios  
- Consider the needs of different user groups

Please output in JSON format with the following structure:  
```json  
[  
  {{
    "name": "Scenario Name",  
    "description": "Detailed description",  
    "context": "Application context",
    "target_users": ["Target user groups"]  
  }}  
]
```
"""