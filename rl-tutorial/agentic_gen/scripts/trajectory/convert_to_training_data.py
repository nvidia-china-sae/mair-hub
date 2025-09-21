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
Training Data Conversion Script

Convert high-quality trajectory data to standard training data format
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager


def setup_conversion_logger():
    """Set up logger for data conversion"""
    logger = setup_logger(
        "training_data_conversion",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    return logger


def load_high_quality_trajectories(source_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load high-quality trajectory data
    
    Args:
        source_dir: High-quality trajectory directory
        logger: Logger instance
        
    Returns:
        List of trajectory data
    """
    logger.info(f"Starting to load high-quality trajectory data: {source_dir}")
    
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return []
    
    # Find all JSON files
    json_files = list(source_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} high-quality trajectory files")
    
    trajectories = []
    failed_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data['_source_file'] = json_file.name
                trajectories.append(data)
            else:
                logger.warning(f"Skipping invalid format file: {json_file.name}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to load file {json_file.name}: {e}")
            failed_count += 1
    
    logger.info(f"Successfully loaded {len(trajectories)} trajectories, failed {failed_count}")
    return trajectories


def load_agents_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Load agent data
    
    Args:
        logger: Logger instance
        
    Returns:
        Agent data dictionary {agent_id: agent_data}
    """
    try:
        agents_dir = settings.get_data_path('agents')
        
        if not agents_dir.exists():
            logger.warning(f"Agent directory does not exist: {agents_dir}")
            return {}
        
        # Find the latest agent file
        agents_files = list(agents_dir.glob('agents_batch_*.json'))
        
        if not agents_files:
            logger.warning("No agent data files found")
            return {}
        
        # Use the latest file
        latest_file = max(agents_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading agent data: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            agents_list = json.load(f)
        
        # Convert to dictionary format
        agents_dict = {}
        for agent in agents_list:
            if isinstance(agent, dict) and 'id' in agent:
                agents_dict[agent['id']] = agent
        
        logger.info(f"Successfully loaded {len(agents_dict)} agents")
        return agents_dict
        
    except Exception as e:
        logger.error(f"Failed to load agent data: {e}")
        return {}


def load_tools_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Load tool data
    
    Args:
        logger: Logger instance
        
    Returns:
        Tool data dictionary {tool_id: tool_data}
    """
    try:
        tools_dir = settings.get_data_path('tools')
        
        if not tools_dir.exists():
            logger.warning(f"Tool directory does not exist: {tools_dir}")
            return {}
        
        # Find tool data files
        tool_files = list(tools_dir.glob('final_tools_*.json'))
        
        if not tool_files:
            logger.warning("No tool data files found")
            return {}
        
        # Use the latest file
        latest_file = max(tool_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading tool data: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
        
        # Handle different data formats
        tools_dict = {}
        if isinstance(tools_data, list):
            # If it's a list format, convert to dictionary
            for tool in tools_data:
                if isinstance(tool, dict) and 'id' in tool:
                    tools_dict[tool['id']] = tool
        elif isinstance(tools_data, dict):
            tools_dict = tools_data
        
        logger.info(f"Successfully loaded {len(tools_dict)} tools")
        return tools_dict
        
    except Exception as e:
        logger.error(f"Failed to load tool data: {e}")
        return {}


def extract_tools_info(
    trajectory_data: Dict[str, Any], 
    agents_data: Dict[str, Any], 
    tools_data: Dict[str, Any],
    logger: logging.Logger
) -> str:
    """
    Extract tool information from trajectory data
    
    Args:
        trajectory_data: Trajectory data
        agents_data: Agent data
        tools_data: Tool data
        logger: Logger instance
        
    Returns:
        Tool description JSON string
    """
    try:
        # 1. Get agent_id from trajectory
        agent_id = trajectory_data.get('agent_id', '')
        
        if not agent_id:
            logger.warning(f"Trajectory {trajectory_data.get('trajectory_id', 'unknown')} has no agent_id")
            return "[]"
        
        # 2. Find corresponding agent data
        if agent_id not in agents_data:
            logger.warning(f"Cannot find agent data: {agent_id}")
            return "[]"
        
        agent_data = agents_data[agent_id]
        
        # 3. Get agent's available tools (tool ids)
        agent_tools = agent_data.get('tools', [])
        
        if not agent_tools:
            logger.warning(f"Agent {agent_id} has no configured tools")
            return "[]"
        
        # 4. Get tool descriptions from tools data based on tool_id
        tools_info = []
        
        for tool_id in agent_tools:
            if tool_id in tools_data:
                tool_data = tools_data[tool_id]
                
                # Convert to standard format
                tool_description = {
                    "name": tool_data.get('name', tool_id),
                    "description": tool_data.get('description', f"Tool {tool_id}"),
                    "parameters": tool_data.get('parameters', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
                
                tools_info.append(tool_description)
            else:
                logger.warning(f"Cannot find tool data: {tool_id}")
        
        return json.dumps(tools_info, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Failed to extract tool information: {e}")
        return "[]"


def extract_json_from_content(content: str) -> str:
    """
    Extract pure JSON string from content (based on tool_execution_simulator.py logic)
    
    Args:
        content: Raw content string
        
    Returns:
        Extracted pure JSON string
    """
    import re
    
    def find_balanced_json(text: str, start_pos: int = 0) -> tuple:
        """
        Find balanced JSON object starting from specified position
        Returns (json_str, end_pos) or (None, -1)
        """
        brace_count = 0
        in_string = False
        escape_next = False
        start_brace_pos = -1
        
        i = start_pos
        while i < len(text):
            char = text[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue
                
            if char == '"':
                in_string = not in_string
                i += 1
                continue
                
            if not in_string:
                if char == '{':
                    if start_brace_pos == -1:
                        start_brace_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_brace_pos != -1:
                        json_candidate = text[start_brace_pos:i+1]
                        try:
                            json.loads(json_candidate)
                            return json_candidate, i+1
                        except json.JSONDecodeError:
                            # Reset and continue searching
                            brace_count = 0
                            start_brace_pos = -1
                            
            i += 1
                            
        return None, -1
    
    try:
        content_str = str(content).strip()
        
        # If content is empty, return directly
        if not content_str:
            return content_str
        
        processed_json_strings = set()
        extracted_json = None
        
        # 1. Extract ```json ... ``` code blocks
        json_code_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_code_pattern, content_str, re.DOTALL)
        for match in matches:
            json_content = match.strip()
            if json_content and json_content not in processed_json_strings:
                processed_json_strings.add(json_content)
                try:
                    # Verify if it's valid JSON
                    json.loads(json_content)
                    extracted_json = json_content
                    break
                except json.JSONDecodeError:
                    continue
        
        # 2. If not found, extract ``` ... ``` code blocks (without language identifier)
        if not extracted_json:
            code_block_pattern = r'```\s*(.*?)\s*```'
            matches_code = re.findall(code_block_pattern, content_str, re.DOTALL)
            for match in matches_code:
                code_content = match.strip()
                if code_content and code_content not in processed_json_strings:
                    processed_json_strings.add(code_content)
                    try:
                        # Verify if it's valid JSON
                        json.loads(code_content)
                        extracted_json = code_content
                        break
                    except json.JSONDecodeError:
                        continue
        
        # 3. If still not found, try to parse entire content as JSON string (for pure JSON format)
        if not extracted_json:
            # Remove all code blocks from content
            remaining_content = content_str
            remaining_content = re.sub(r'```json.*?```', ' ', remaining_content, flags=re.DOTALL)
            remaining_content = re.sub(r'```.*?```', ' ', remaining_content, flags=re.DOTALL)
            remaining_content = remaining_content.strip()
            
            # Try to parse entire content directly
            if remaining_content and remaining_content not in processed_json_strings:
                try:
                    # Verify if it's valid JSON
                    json.loads(remaining_content)
                    extracted_json = remaining_content
                except json.JSONDecodeError:
                    pass
        
        # 4. If still not found, use balanced search to find JSON objects
        if not extracted_json:
            remaining_content = content_str
            # Remove all code blocks
            remaining_content = re.sub(r'```json.*?```', ' ', remaining_content, flags=re.DOTALL)
            remaining_content = re.sub(r'```.*?```', ' ', remaining_content, flags=re.DOTALL)
            
            # Find the first complete JSON object
            pos = 0
            while pos < len(remaining_content):
                json_candidate, next_pos = find_balanced_json(remaining_content, pos)
                if json_candidate and json_candidate not in processed_json_strings:
                    processed_json_strings.add(json_candidate)
                    # Verify if JSON contains name field (prioritize JSON with name field)
                    try:
                        parsed = json.loads(json_candidate)
                        if isinstance(parsed, dict) and 'name' in parsed:
                            extracted_json = json_candidate
                            break
                        elif not extracted_json:  # If no JSON found yet, save current one
                            extracted_json = json_candidate
                    except json.JSONDecodeError:
                        pass
                    pos = next_pos
                else:
                    pos += 1
        
        # If extracted JSON is found, return it, otherwise return original content
        return extracted_json if extracted_json else content_str
        
    except Exception as e:
        # If error occurs during processing, return original content
        return str(content)


def convert_trajectory_to_training_format(
    trajectory_data: Dict[str, Any], 
    agents_data: Dict[str, Any], 
    tools_data: Dict[str, Any],
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Convert single trajectory to training data format
    
    Args:
        trajectory_data: Trajectory data
        agents_data: Agent data
        tools_data: Tool data
        logger: Logger instance
        
    Returns:
        Converted training data
    """
    try:
        # Extract message list
        messages = trajectory_data.get('messages', [])
        if not messages:
            logger.warning(f"Trajectory {trajectory_data.get('trajectory_id', 'unknown')} has no message data")
            return None
        
        # Remove last human message (not needed for training)
        if messages and messages[-1].get('role') == 'user':
            messages = messages[:-1]
            logger.debug(f"Removed last human message: {trajectory_data.get('trajectory_id', 'unknown')}")
        
        conversations = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            recipient = message.get('recipient', '')
            
            # Role mapping
            if role == 'user':
                conversations.append({
                    "from": "human",
                    "value": str(content)
                })
            elif role == 'assistant':
                # Check if it contains tool calls
                content_str = str(content)
                if recipient == 'execution':
                    # This is a tool call, extract pure JSON format
                    clean_json = extract_json_from_content(content_str)
                    
                    # Verify if JSON contains necessary fields
                    try:
                        parsed_json = json.loads(clean_json)
                        if not isinstance(parsed_json, dict):
                            logger.warning(f"Trajectory {trajectory_data.get('trajectory_id', 'unknown')} function_call is not a valid JSON object")
                            return None
                        
                        # Check if it contains name and arguments fields
                        if 'name' not in parsed_json or 'arguments' not in parsed_json:
                            logger.warning(f"Trajectory {trajectory_data.get('trajectory_id', 'unknown')} function_call missing name or arguments fields")
                            return None
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Trajectory {trajectory_data.get('trajectory_id', 'unknown')} function_call is not valid JSON format")
                        return None
                    
                    conversations.append({
                        "from": "function_call",
                        "value": clean_json
                    })
                else:
                    # This is a normal assistant reply
                    conversations.append({
                        "from": "gpt",
                        "value": content_str
                    })
            elif role == 'execution':
                # Tool execution results
                if isinstance(content, list):
                    # Merge multiple execution results
                    observation_content = []
                    for result in content:
                        if isinstance(result, dict):
                            result.pop('metadata')
                            observation_content.append(json.dumps(result, ensure_ascii=False))
                        else:
                            print(result)
                            observation_content.append(str(result))
                    conversations.append({
                        "from": "observation", 
                        "value": "\n".join(observation_content)
                    })
                else:
                    conversations.append({
                        "from": "observation",
                        "value": str(content)
                    })
        
        # If no valid conversations, skip
        if not conversations:
            logger.warning(f"Trajectory {trajectory_data.get('trajectory_id', 'unknown')} has no valid conversations")
            return None
        
        # Extract tool information
        tools_info = extract_tools_info(trajectory_data, agents_data, tools_data, logger)
        
        training_item = {
            "conversations": conversations,
            "tools": tools_info
        }
        
        return training_item
        
    except Exception as e:
        logger.error(f"Failed to convert trajectory {trajectory_data.get('trajectory_id', 'unknown')}: {e}")
        return None


def convert_trajectories_to_training_data(
    trajectories: List[Dict[str, Any]], 
    agents_data: Dict[str, Any], 
    tools_data: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Batch convert trajectories to training data
    
    Args:
        trajectories: List of trajectory data
        agents_data: Agent data
        tools_data: Tool data
        logger: Logger instance
        
    Returns:
        List of training data
    """
    logger.info(f"Starting to convert {len(trajectories)} trajectories to training data")
    
    training_data = []
    success_count = 0
    failed_count = 0
    
    for i, trajectory in enumerate(trajectories, 1):
        try:
            training_item = convert_trajectory_to_training_format(trajectory, agents_data, tools_data, logger)
            if training_item:
                training_data.append(training_item)
                success_count += 1
            else:
                failed_count += 1
            
            # Progress indicator
            if i % 10 == 0:
                logger.info(f"Conversion progress: {i}/{len(trajectories)} ({i/len(trajectories)*100:.1f}%)")
        
        except Exception as e:
            failed_count += 1
            logger.error(f"Error converting trajectory {i}: {e}")
    
    logger.info(f"Conversion completed: {success_count} successful, {failed_count} failed")
    return training_data


def save_training_data(
    training_data: List[Dict[str, Any]], 
    target_dir: Path, 
    logger: logging.Logger
) -> str:
    """
    Save training data
    
    Args:
        training_data: List of training data
        target_dir: Target directory
        logger: Logger instance
        
    Returns:
        Saved file path
    """
    try:
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{timestamp}.json"
        
        file_path = target_dir / filename
        
        # Save data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training data saved: {file_path}")
        logger.info(f"Total training samples: {len(training_data)}")
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")
        raise


def print_conversion_summary(
    original_count: int, 
    converted_count: int, 
    output_file: str
):
    """Print conversion result summary"""
    print(f"\n📊 Training Data Conversion Results")
    print("="*60)
    
    print(f"📈 Conversion Statistics:")
    print(f"  Original trajectories: {original_count}")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Conversion success rate: {converted_count/original_count*100:.1f}%" if original_count > 0 else "  Conversion success rate: 0.0%")
    
    print(f"\n💾 Output File:")
    print(f"  File path: {output_file}")
    print(f"  Training samples: {converted_count}")
    
    if converted_count > 0:
        print(f"\n✅ Training data preparation completed!")
        print(f"📝 Data format: Each sample contains conversations and tools fields")
        print(f"🔧 Conversation roles: human, gpt, function_call, observation")
    else:
        print(f"\n❌ No successfully converted training data")


def main():
    """Main function"""
    print("🔄 Training Data Converter")
    print("="*60)
    
    # Setup logging
    logger = setup_conversion_logger()
    
    try:
        # 1. Setup directory paths
        source_dir = settings.get_data_path('high_quality_trajectories')
        target_dir = settings.get_data_path('training_data')
        
        print(f"📁 Source directory: {source_dir}")
        print(f"📁 Target directory: {target_dir}")
        
        # 2. Load high-quality trajectory data
        print("📂 Loading high-quality trajectory data...")
        trajectories = load_high_quality_trajectories(source_dir, logger)
        
        if not trajectories:
            print("❌ No high-quality trajectory data found")
            return 1
        
        print(f"✅ Successfully loaded {len(trajectories)} high-quality trajectories")
        
        # 3. Load agent data
        print("📂 Loading agent data...")
        agents_data = load_agents_data(logger)
        
        if not agents_data:
            print("⚠️ No agent data loaded, will use default tool configuration")
        else:
            print(f"✅ Successfully loaded {len(agents_data)} agents")
        
        # 4. Load tool data
        print("🔧 Loading tool data...")
        tools_data = load_tools_data(logger)
        
        if not tools_data:
            print("⚠️ No tool data loaded, will use default tool definitions")
        else:
            print(f"✅ Successfully loaded {len(tools_data)} tools")
        
        # 5. Convert to training data format
        print("🔄 Converting trajectory data to training format...")
        training_data = convert_trajectories_to_training_data(trajectories, agents_data, tools_data, logger)
        
        if not training_data:
            print("❌ No successfully converted training data")
            return 1
        
        # 6. Save training data
        print("💾 Saving training data...")
        output_file = save_training_data(training_data, target_dir, logger)
        
        # 7. Display conversion result summary
        print_conversion_summary(len(trajectories), len(training_data), output_file)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
        return 1
    except Exception as e:
        logger.error(f"Training data conversion failed: {e}")
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
