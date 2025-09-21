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

#!/usr/bin/env python3
"""
Task Generation Script

Batch generate multi-turn conversation tasks for agents, including detailed scoring checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
import random

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager
from modules.task_generator import TaskGenerator


def setup_task_logger():
    """Setup task generation specific logger"""
    logger = setup_logger(
        "task_generation",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    return logger

def find_latest_agents_file() -> str:
    """Find the latest agents file"""
    data_path = Path('data/generated/agents')
    if not data_path.exists():
        raise FileNotFoundError("Agents data directory not found")
    
    # Prioritize searching for agents_batch files
    agents_files = list(data_path.glob('agents_batch_*.json'))
    
    if not agents_files:
        raise FileNotFoundError("No agents batch files found")
    
    # Return the latest file
    latest_file = max(agents_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)


def find_latest_tools_file() -> str:
    """Find the latest tools file"""
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Prioritize searching for final filtered tools files
    final_files = file_manager.list_files(".", "*final_tools*.json")
    if final_files:
        latest_file = max(final_files, key=lambda f: file_manager.get_file_info(f)['modified'])
        return os.path.join(tools_dir, latest_file)
    
    return None


def load_agents_data(file_path: str) -> List[Dict[str, Any]]:
    """Load agents data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'agents' in data:
        return data['agents']
    else:
        raise ValueError("Invalid agents data format")


def load_tools_data(file_path: str) -> Dict[str, Any]:
    """Load tools data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Convert to dictionary format {tool_id: tool_data}
        return {tool['id']: tool for tool in data}
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError("Invalid tools data format")


def validate_agent_tools(agents: List[Dict[str, Any]], tools_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate the validity of agent tools"""
    valid_agents = []
    
    for agent in agents:
        agent_tools = agent.get('tools', [])
        valid_tools = []
        
        for tool_id in agent_tools:
            if tool_id in tools_data:
                valid_tools.append(tool_id)
            else:
                print(f"Warning: Tool {tool_id} not found for agent {agent.get('id')}")
        
        if len(valid_tools) >= 2:  # At least 2 tools are required to generate multi-turn tasks
            agent['tools'] = valid_tools
            valid_agents.append(agent)
        else:
            print(f"Warning: Agent {agent.get('id')} has insufficient valid tools ({len(valid_tools)})")
    
    return valid_agents


def main():
    """Main function"""
    # Setup logging
    logger = setup_task_logger()
    
    try:
        print("🎯 Starting task generation process...")
        
        # 1. Find data files
        print("📁 Finding latest data files...")
        agents_file = find_latest_agents_file()
        tools_file = find_latest_tools_file()
        
        print(f"Agents file: {agents_file}")
        print(f"Tools file: {tools_file}")
        
        # 2. Load data
        print("📊 Loading data...")
        agents_data = load_agents_data(agents_file)
        tools_data = load_tools_data(tools_file)
        
        print(f"Loaded {len(agents_data)} agents")
        print(f"Loaded {len(tools_data)} tools")
        
        # 3. Validate data
        print("✅ Validating agent tools validity...")
        valid_agents = validate_agent_tools(agents_data, tools_data)
        print(f"Valid agents count: {len(valid_agents)}")
        
        if not valid_agents:
            print("❌ No valid agents found, cannot generate tasks")
            return
        
        random.shuffle(valid_agents)
        # 5. Configure task generation
        task_config = settings.GENERATION_CONFIG.get('tasks', {})
        max_workers = settings.CONCURRENCY_CONFIG.get('max_workers', 4)
        task_config['max_workers'] = max_workers
        
        # 6. Initialize task generation module
        print("🚀 Initializing task generation module...")
        task_generator = TaskGenerator(config=task_config, logger=logger)
        task_generator.initialize()
        
        # 7. Generate tasks
        print("🎨 Starting task generation...")
        result = task_generator.process({
            'agents': valid_agents,
            'tools_data': tools_data
        })
        
        # 8. Display results
        total_tasks = result['total_tasks']
        total_agents = result['total_agents']
        
        print(f"\\n✅ Task generation completed!")
        print(f"Processed agents count: {total_agents}")
        print(f"Total tasks generated: {total_tasks}")
        
        # Display difficulty distribution and success rate
        difficulty_dist = result['generation_summary']['difficulty_distribution']
        success_rate = result['generation_summary']['success_rate']
        
        print(f"\\n📊 Task difficulty distribution:")
        for difficulty, count in difficulty_dist.items():
            print(f"  {difficulty}: {count} tasks")
        
        print(f"\\n📈 Generation success rate: {success_rate:.2%}")
        print(f"\\n💾 Task data has been saved to data/generated/tasks/ directory")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please ensure agents and tools data have been generated")
    except Exception as e:
        print(f"❌ Task generation failed: {e}")
        logger.error(f"Task generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
