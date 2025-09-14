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
Trajectory Generation Script
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager
from core.models import Task, AgentConfig, TaskRubric, DifficultyLevel, TaskType
from modules.interaction_coordinator import InteractionCoordinator


def setup_trajectory_logger():
    """Set up logger for trajectory generation"""
    logger = setup_logger(
        "trajectory_generation",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    return logger


def validate_environment():
    """Validate environment configuration"""
    required_keys = ['OPENAI_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing environment variables: {', '.join(missing_keys)}")
        print("Please ensure the following variables are set in the .env file:")
        for key in missing_keys:
            print(f"  {key}=your_api_key_here")
        return False
    
    print("✅ Environment variable check passed")
    return True


def find_latest_tasks_file() -> Optional[str]:
    """Find the latest tasks file"""
    tasks_dir = settings.get_data_path('tasks')
    file_manager = FileManager(tasks_dir)
    
    # Find batch task files
    batch_files = file_manager.list_files(".", "*tasks_batch*.json")
    if batch_files:
        latest_file = max(batch_files, key=lambda f: file_manager.get_file_info(f)['modified'])
        return os.path.join(tasks_dir, latest_file)
    
    return None


def find_latest_agents_file() -> Optional[str]:
    """Find the latest agents file"""
    agents_dir = settings.get_data_path('agents')
    
    if not agents_dir.exists():
        return None
    
    # Prioritize agents_batch files
    agents_files = list(agents_dir.glob('agents_batch_*.json'))
    
    if agents_files:
        latest_file = max(agents_files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    
    return None


def find_latest_tools_file() -> Optional[str]:
    """Find the latest tools file"""
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Prioritize final filtered tool files
    final_files = file_manager.list_files(".", "*final_tools*.json")
    if final_files:
        latest_file = max(final_files, key=lambda f: file_manager.get_file_info(f)['modified'])
        return os.path.join(tools_dir, latest_file)
    
    return None


def load_tasks_data(file_path: str) -> List[Dict[str, Any]]:
    """Load task data"""
    print(f"📂 Loading task data: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tasks_data = json.load(f)
    
    if not isinstance(tasks_data, list):
        raise ValueError("Invalid tasks data format: expected list")
    
    print(f"✅ Successfully loaded {len(tasks_data)} tasks")
    return tasks_data


def load_agents_data(file_path: str) -> List[Dict[str, Any]]:
    """Load agent data"""
    print(f"📂 Loading agent data: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        agents_data = data
    elif isinstance(data, dict) and 'agents' in data:
        agents_data = data['agents']
    else:
        raise ValueError("Invalid agents data format")
    
    print(f"✅ Successfully loaded {len(agents_data)} agents")
    return agents_data


def load_tools_data(file_path: str) -> Dict[str, Any]:
    """Load tool data"""
    print(f"📂 Loading tool data: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Convert to dictionary format {tool_id: tool_data}
        tools_data = {tool['id']: tool for tool in data}
    elif isinstance(data, dict):
        tools_data = data
    else:
        raise ValueError("Invalid tools data format")
    
    print(f"✅ Successfully loaded {len(tools_data)} tools")
    return tools_data


def convert_task_dict_to_object(task_data: Dict[str, Any]) -> Task:
    """Convert task dictionary to Task object"""
    # Convert TaskRubric
    rubric_data = task_data['rubric']
    rubric = TaskRubric(
        success_criteria=rubric_data['success_criteria'],
        tool_usage_expectations=rubric_data.get('tool_usage_expectations', []),
        checkpoints=rubric_data['checkpoints']
    )
    
    # Map difficulty and task_type
    difficulty_map = {
        'simple': DifficultyLevel.SIMPLE,
        'medium': DifficultyLevel.MEDIUM,
        'complex': DifficultyLevel.COMPLEX
    }
    
    task_type_map = {
        'multi_turn': TaskType.MULTI_TURN,
        'single_turn': TaskType.SINGLE_TURN
    }
    
    task = Task(
        id=task_data['id'],
        agent_id=task_data['agent_id'],
        title=task_data['title'],
        description=task_data['description'],
        difficulty=difficulty_map.get(task_data['difficulty'], DifficultyLevel.MEDIUM),
        task_type=task_type_map.get(task_data['task_type'], TaskType.MULTI_TURN),
        expected_tools=task_data['expected_tools'],
        rubric=rubric,
        metadata=task_data.get('metadata', {})
    )
    
    return task


def convert_agent_dict_to_object(agent_data: Dict[str, Any]) -> AgentConfig:
    """Convert agent dictionary to AgentConfig object"""
    return AgentConfig(
        id=agent_data['id'],
        system_prompt=agent_data['system_prompt'],
        tools=agent_data['tools']
    )


def load_existing_trajectory_task_ids(trajectory_dir: Path, logger: logging.Logger) -> set:
    """
    Load task IDs from existing trajectories
    
    Args:
        trajectory_dir: Trajectory directory path
        logger: Logger instance
        
    Returns:
        Set of existing task IDs
    """
    existing_task_ids = set()
    
    if not trajectory_dir.exists():
        logger.info(f"Trajectory directory does not exist: {trajectory_dir}")
        return existing_task_ids
    
    # Find all JSON files
    json_files = list(trajectory_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} trajectory files in {trajectory_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            
            # Extract task_id
            task_id = trajectory_data.get('task_id')
            if task_id:
                existing_task_ids.add(task_id)
                logger.debug(f"Found existing task ID: {task_id} (from file: {json_file.name})")
            
        except Exception as e:
            logger.warning(f"Failed to read trajectory file {json_file.name}: {e}")
            continue
    
    logger.info(f"Total found {len(existing_task_ids)} existing task IDs")
    return existing_task_ids


def filter_existing_tasks(
    matched_pairs: List[Tuple[Task, AgentConfig, Dict[str, Any]]], 
    existing_task_ids: set,
    logger: logging.Logger
) -> List[Tuple[Task, AgentConfig, Dict[str, Any]]]:
    """
    Filter out tasks that already exist
    
    Args:
        matched_pairs: List of matched task-agent pairs
        existing_task_ids: Set of existing task IDs
        logger: Logger instance
        
    Returns:
        Filtered list of matched pairs
    """
    if not existing_task_ids:
        logger.info("No existing tasks found, no filtering needed")
        return matched_pairs
    
    filtered_pairs = []
    filtered_count = 0
    
    for task, agent, tools in matched_pairs:
        if task.id in existing_task_ids:
            logger.debug(f"Filtering existing task: {task.id}")
            filtered_count += 1
        else:
            filtered_pairs.append((task, agent, tools))
    
    logger.info(f"Filtered out {filtered_count} existing tasks, {len(filtered_pairs)} tasks remaining for generation")
    return filtered_pairs


def match_tasks_and_agents(tasks_data: List[Dict[str, Any]], 
                          agents_data: List[Dict[str, Any]], 
                          tools_data: Dict[str, Any]) -> List[Tuple[Task, AgentConfig, Dict[str, Any]]]:
    """Match tasks and agents, and verify tool availability"""
    matched_pairs = []
    agents_dict = {agent['id']: agent for agent in agents_data}
    
    print("🔗 Matching tasks and agents...")
    
    for task_data in tasks_data:
        agent_id = task_data['agent_id']
        
        # Find corresponding agent
        if agent_id not in agents_dict:
            print(f"⚠️ Agent {agent_id} for task {task_data['id']} not found, skipping")
            continue
        
        agent_data = agents_dict[agent_id]
        agent_tools = agent_data.get('tools', [])
        
        # Verify tool availability
        available_tools = {}
        valid_tools_count = 0
        
        for tool_id in agent_tools:
            if tool_id in tools_data:
                available_tools[tools_data[tool_id]['name']] = tools_data[tool_id]
                valid_tools_count += 1
        
        if valid_tools_count < 2:  # At least 2 valid tools required
            print(f"⚠️ Agent {agent_id} has insufficient valid tools ({valid_tools_count}), skipping")
            continue
        
        # Convert to objects
        task_obj = convert_task_dict_to_object(task_data)
        agent_obj = convert_agent_dict_to_object(agent_data)
        
        matched_pairs.append((task_obj, agent_obj, available_tools))
    
    print(f"✅ Successfully matched {len(matched_pairs)} task-agent pairs")
    return matched_pairs


def generate_single_trajectory(logger: logging.Logger,
                             task: Task, 
                             agent_config: AgentConfig, 
                             tools_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate a single trajectory (each thread uses independent coordinator)"""
    try:
        # Create independent coordinator instance for each trajectory generation to avoid concurrency issues
        trajectory_config = settings.GENERATION_CONFIG.get('trajectories', {})
        coordinator = InteractionCoordinator(config=trajectory_config, logger=logger)
        coordinator.initialize()
        
        trajectory = coordinator.execute_single_interaction(task, agent_config, tools_info)
        
        return {
            'trajectory_id': trajectory.id,
            'task_id': task.id,
            'agent_id': agent_config.id,
            'turns_count': len(trajectory.session.turns),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Failed to generate trajectory - Task: {task.id}, Agent: {agent_config.id}, Error: {e}")
        return {
            'task_id': task.id,
            'agent_id': agent_config.id,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main function"""
    print("🎯 Trajectory Generator")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
    # Setup logging
    logger = setup_trajectory_logger()
    
    try:
        # 1. Find data files
        print("📁 Finding latest data files...")
        
        tasks_file = find_latest_tasks_file()
        agents_file = find_latest_agents_file()
        tools_file = find_latest_tools_file()
        
        if not tasks_file:
            print("❌ Task data file not found")
            return
        if not agents_file:
            print("❌ Agent data file not found")
            return
        if not tools_file:
            print("❌ Tool data file not found")
            return
        
        print(f"Task file: {os.path.basename(tasks_file)}")
        print(f"Agent file: {os.path.basename(agents_file)}")
        print(f"Tool file: {os.path.basename(tools_file)}")
        
        # 2. Load data
        print("\n📊 Loading data...")
        tasks_data = load_tasks_data(tasks_file)
        agents_data = load_agents_data(agents_file)
        tools_data = load_tools_data(tools_file)
        
        # 3. Match tasks and agents
        print("\n🔗 Matching data...")
        matched_pairs = match_tasks_and_agents(tasks_data, agents_data, tools_data)
        
        # 4. Filter existing tasks
        print("\n🔍 Checking and filtering existing tasks...")
        trajectory_dir = settings.DATA_DIR / "generated" / "trajectories"
        existing_task_ids = load_existing_trajectory_task_ids(trajectory_dir, logger)
        
        if existing_task_ids:
            print(f"Found {len(existing_task_ids)} existing tasks, will filter them out")
            matched_pairs = filter_existing_tasks(matched_pairs, existing_task_ids, logger)
        else:
            print("No existing tasks found")
        
        if not matched_pairs:
            print("❌ After filtering, no task-agent pairs found for generation")
            return
        
        # 5. Get configuration
        trajectory_config = settings.GENERATION_CONFIG.get('trajectories', {})
        max_trajectories = trajectory_config.get('max_count', 10)  # Limit generation count
        max_workers = trajectory_config.get('max_workers', 8)
        
        # Randomly select pairs (avoid generating all)
        if len(matched_pairs) > max_trajectories:
            print(f"🎲 Randomly selecting {max_trajectories} pairs for generation")
            matched_pairs = random.sample(matched_pairs, max_trajectories)
        
        print(f"\n🎯 Generation configuration:")
        print(f"  Target trajectory count: {len(matched_pairs)}")
        print(f"  Concurrency: {max_workers}")

        # 6. Prepare trajectory generation
        print("\n🔄 Starting trajectory generation...")

        start_time = datetime.now()
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Use multi-threading to generate trajectories
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {}
            
            for task, agent_config, tools_info in matched_pairs:
                future = executor.submit(
                    generate_single_trajectory,
                    logger,
                    task,
                    agent_config,
                    tools_info
                )
                future_to_params[future] = (task.id, agent_config.id)
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_params), 1):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        successful_count += 1
                        if successful_count % 10 == 0:  # Output progress every 10 successful trajectories
                            print(f"✅ Successfully generated {successful_count} trajectories...")
                    else:
                        failed_count += 1
                        
                    # Output total progress
                    if i % 20 == 0:
                        print(f"📊 Total progress: {i}/{len(matched_pairs)} ({i/len(matched_pairs)*100:.1f}%)")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Trajectory generation task exception: {e}")
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # 7. Output statistics
        print(f"\n✅ Trajectory generation completed!")
        print(f"📊 Generation statistics:")
        print(f"  Successfully generated: {successful_count} trajectories")
        print(f"  Failed count: {failed_count}")
        print(f"  Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
        print(f"  Total time: {generation_time:.2f} seconds")
        
        if successful_count > 0:
            print(f"  Generation speed: {successful_count/generation_time:.1f} trajectories/second")
        
        print(f"\n💾 Trajectory data saved to data/generated/trajectories/ directory")
        
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
    except Exception as e:
        logger.error(f"Trajectory generation failed: {e}")
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
