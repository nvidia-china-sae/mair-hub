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
Agent generation script
Generates diverse agent configurations based on tool data
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from modules.agent_synthesizer import AgentSynthesizerModule
from utils.logger import setup_logger
from utils.file_manager import FileManager


def setup_agent_logger():
    """Setup dedicated logger for agent generation"""
    logger = setup_logger(
        "agent_generation",
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


def find_latest_tools_file():
    """Find the latest tools file"""
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Prioritize finding final filtered tools file
    final_files = file_manager.list_files(".", "*final_tools*.json")
    if final_files:
        latest_file = max(final_files, key=lambda f: file_manager.get_file_info(f)['modified'])
        return os.path.join(tools_dir, latest_file)
    
    return None


def load_tools_data(file_path: str):
    """Load tool data"""
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return []
    
    print(f"📂 Loading tool data: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    
    print(f"✅ Successfully loaded {len(tools_data)} tools")
    
    return tools_data

def main():
    """Main function"""
    print("🤖 Agent Generator")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
    # Setup logging
    logger = setup_agent_logger()
    
    try:
        # 1. Automatically find latest tools file
        tools_file = find_latest_tools_file()
        if not tools_file:
            print("❌ No tool data file found")
            return
        
        # 2. Load tool data
        tools_data = load_tools_data(tools_file)
        if not tools_data:
            print("❌ Unable to load tool data, program exiting")
            return
        
        # 3. Get configuration
        agent_config = settings.GENERATION_CONFIG.get('agents', {})
        target_count = agent_config.get('target_count', 1000)
        
        print(f"\n🎯 Generation configuration:")
        print(f"  Target agent count: {target_count}")
        print(f"  Total tools: {len(tools_data)}")
        
        tools_per_agent = agent_config.get('tools_per_agent', {})
        min_tools = tools_per_agent.get('min', 3)
        max_tools = tools_per_agent.get('max', 6)
        print(f"  Tools per agent: {min_tools}-{max_tools}")
        
        # 4. Initialize agent synthesis module
        print("\n⚙️ Initializing agent synthesis module...")
        synthesizer = AgentSynthesizerModule(logger=logger)
        synthesizer.initialize()
        
        # 5. Generate agent configurations
        print("\n🔄 Starting agent synthesis...")
        
        start_time = datetime.now()
        
        result = synthesizer.process({
            'tools': tools_data,
            'target_agent_count': target_count
        })
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️ Generation time: {generation_time:.2f} seconds")
        print(f"📊 Generation speed: {len(result.get('agents', []))/generation_time:.1f} agents/second")
        
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
    except Exception as e:
        logger.error(f"Agent generation failed: {e}")
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
