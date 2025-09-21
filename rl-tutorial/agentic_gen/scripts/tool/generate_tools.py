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
Tool Generation Script
Generate large amounts of tools based on generated scenario data
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager
from modules.domain_tool_generator.tool_designer import ToolDesigner


def setup_tool_logger():
    """Setup tool generation dedicated logger"""
    logger = setup_logger(
        "tool_generation",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"],
        format_string=settings.LOGGING_CONFIG["format"]
    )
    return logger


def validate_environment(logger):
    """Validate runtime environment"""
    logger.info("Validating environment configuration...")
    
    # Check API key
    llm_config = settings.get_llm_config()
    if not llm_config.get("api_key"):
        logger.error(f"Missing {settings.DEFAULT_LLM_PROVIDER} API key")
        logger.error("Please set environment variable: OPENAI_API_KEY or CLAUDE_API_KEY")
        return False
    
    # Check if scenario data exists
    scenarios_path = settings.get_data_path('scenarios')
    if not scenarios_path.exists() or not any(scenarios_path.glob("*.json")):
        logger.error(f"Scenario data files not found, please run generate_scenarios.py first")
        logger.error(f"Scenario data path: {scenarios_path}")
        return False
    
    # Check and create tool data directory
    tools_path = settings.get_data_path('tools')
    if not tools_path.exists():
        tools_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created tool data directory: {tools_path}")
    
    logger.info("Environment validation completed")
    return True


def load_existing_scenarios(logger):
    """Load generated scenario data"""
    logger.info("Loading generated scenario data...")
    
    scenarios_path = settings.get_data_path('scenarios')
    file_manager = FileManager(scenarios_path, logger)
    
    # Find latest scenario file
    scenario_files = list(scenarios_path.glob("scenarios_batch_*.json"))
    if not scenario_files:
        all_scenario_files = list(scenarios_path.glob("all_scenarios_*.json"))
        if all_scenario_files:
            latest_file = max(all_scenario_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Using summary scenario file: {latest_file.name}")
            scenarios = file_manager.load_json(latest_file.name)
        else:
            raise FileNotFoundError("Scenario data files not found")
    else:
        latest_file = max(scenario_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using scenario batch file: {latest_file.name}")
        scenarios = file_manager.load_json(latest_file.name)
    
    logger.info(f"Successfully loaded {len(scenarios)} scenarios")
    return scenarios


def generate_tools_for_scenarios(scenarios: List[Dict[str, Any]], logger):
    """Generate tools based on scenarios"""
    logger.info("Starting tool generation based on scenarios...")
    
    try:
        # Get tool generation configuration
        tool_config = settings.GENERATION_CONFIG['tools']
        
        # Get concurrency configuration
        concurrency_config = settings.CONCURRENCY_CONFIG
        max_workers = concurrency_config.get('max_workers', 4)
        
        input_data = {
            'scenarios': scenarios,
        }
        
        designer_config = {
            'batch_size': tool_config.get('batch_size', 20),
            'tools_per_scenario': tool_config.get('tools_per_scenario', 8),
            'max_workers': max_workers,
        }
                
        with ToolDesigner(designer_config, logger) as designer:
            tools = designer.process(input_data)
            designer_stats = designer.get_generation_stats() if hasattr(designer, 'get_generation_stats') else {}
        
        return tools, designer_stats
        
    except Exception as e:
        logger.error(f"Tool generation failed: {e}")
        raise


def analyze_generation_results(scenarios: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                             designer_stats: Dict[str, Any], registration_result: Dict[str, Any], logger):
    """Analyze tool generation results"""
    logger.info("Analyzing tool generation results...")
    
    # Basic statistics
    total_scenarios = len(scenarios)
    total_tools = len(tools)
    tools_per_scenario = total_tools / total_scenarios if total_scenarios > 0 else 0
    
    # Statistics by domain
    domain_stats = {}
    category_stats = {}
    
    for tool in tools:
        metadata = tool.get('metadata', {})
        domain = metadata.get('domain', 'unknown')
        category = metadata.get('category', 'unknown')
        
        domain_stats[domain] = domain_stats.get(domain, 0) + 1
        category_stats[category] = category_stats.get(category, 0) + 1
    
    # Tool type statistics
    tool_types = {}
    for tool in tools:
        tool_type = tool.get('category', 'unknown')
        tool_types[tool_type] = tool_types.get(tool_type, 0) + 1
    
    analysis = {
        'generation_summary': {
            'total_scenarios_used': total_scenarios,
            'total_tools_generated': total_tools,
            'tools_per_scenario_avg': round(tools_per_scenario, 2),
            'registration_result': registration_result
        },
        'domain_distribution': domain_stats,
        'category_distribution': category_stats,
        'tool_type_distribution': tool_types,
        'designer_stats': designer_stats
    }
    
    # Output result summary
    logger.info("=" * 60)
    logger.info("Tool Generation Result Summary")
    logger.info("=" * 60)
    logger.info(f"📊 Total scenarios: {total_scenarios}")
    logger.info(f"🔧 Total tools: {total_tools}")
    logger.info(f"📈 Average tools per scenario: {tools_per_scenario:.2f}")
    logger.info(f"✅ Successfully registered tools: {registration_result.get('registered_count', 0)}")
    
    logger.info("\n📂 Domain distribution:")
    for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"   {domain}: {count} tools")
    
    logger.info("\n🔨 Tool type distribution:")
    for tool_type, count in sorted(tool_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"   {tool_type}: {count} tools")
    
    return analysis


def main():
    """Main function"""
    print("🔧 Tool Generator")
    print("=" * 50)
    
    # Setup logging
    logger = setup_tool_logger()
    
    try:
        # Validate environment
        if not validate_environment(logger):
            return 1
        
        # Load scenario data
        scenarios = load_existing_scenarios(logger)
        # Generate tools
        tools, designer_stats = generate_tools_for_scenarios(scenarios, logger)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 Tool generation completed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Tool generation process failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 