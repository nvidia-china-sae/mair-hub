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
Tool Embedding Vector Calculation Script
Read the latest tool JSON file, calculate embedding vectors for each tool and save results
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager
from modules.domain_tool_generator.tool_embedding import ToolEmbedding


def setup_embedding_logger():
    """Setup embedding vector calculation dedicated logger"""
    logger = setup_logger(
        "tool_embedding",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"],
        format_string=settings.LOGGING_CONFIG["format"]
    )
    return logger


def validate_environment(logger):
    """Validate runtime environment"""
    logger.info("Validating environment configuration...")
    
    # Check embedding configuration
    embedding_config = settings.get_embedding_config()
    api_key = embedding_config.get('api_key')
    
    if not api_key:
        logger.error("Missing DashScope API key")
        logger.error("Please set environment variable: DASHSCOPE_API_KEY")
        return False

    return True


def find_latest_tools_file(logger):
    """Find the latest tool data file"""
    logger.info("Finding latest tool data file...")
    
    tools_path = settings.get_data_path('tools')
    file_manager = FileManager(tools_path, logger)
    
    # Prioritize optimized tool files
    batch_files = list(tools_path.glob("tools_batch_*.json"))
    if batch_files:
        latest_file = max(batch_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using optimized tool file: {latest_file.name}")
        return latest_file
    
    raise FileNotFoundError("No tool data files found")


def load_tools_data(file_path: Path, logger):
    """Load tool data"""
    logger.info(f"Loading tool data: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        logger.info(f"Successfully loaded {len(tools)} tools")
        return tools
        
    except Exception as e:
        logger.error(f"Failed to load tool data: {e}")
        raise


def compute_embeddings_for_tools(tools: List[Dict[str, Any]], logger):
    """Calculate embedding vectors for tools"""
    logger.info("Starting tool embedding vector calculation...")
    
    try:
        input_data = {'tools': tools}
        
        with ToolEmbedding(logger) as embedding_module:
            # Process tool list directly instead of loading from file
            updated_tools = embedding_module._add_embeddings_to_tools(tools)
            
        return updated_tools
        
    except Exception as e:
        logger.error(f"Embedding calculation failed: {e}")
        raise


def merge_with_existing_tools(new_tools: List[Dict[str, Any]], 
                             all_tools: List[Dict[str, Any]], logger):
    """Merge newly calculated embeddings into all tools"""
    logger.info("Merging embedding results...")
    
    # Create mapping from new tool ID to tool
    new_tools_map = {tool['id']: tool for tool in new_tools}
    
    # Update existing tool list
    updated_tools = []
    for tool in all_tools:
        tool_id = tool.get('id')
        if tool_id in new_tools_map:
            # Use newly calculated embedding
            updated_tools.append(new_tools_map[tool_id])
        else:
            # Keep original tool (may already have embedding)
            updated_tools.append(tool)
    
    # Calculate embedding coverage
    tools_with_embedding = len([
        t for t in updated_tools 
        if t.get('metadata', {}).get('embedding')
    ])
    
    logger.info(f"Merge completed:")
    logger.info(f"  Total tools: {len(updated_tools)}")
    logger.info(f"  Tools with embedding: {tools_with_embedding}")
    logger.info(f"  Embedding coverage: {tools_with_embedding/len(updated_tools)*100:.1f}%")
    
    return updated_tools


def analyze_embedding_results(tools: List[Dict[str, Any]], logger):
    """Analyze embedding calculation results"""
    logger.info("Analyzing embedding calculation results...")
    
    # Basic statistics
    total_tools = len(tools)
    tools_with_embedding = len([
        t for t in tools 
        if t.get('metadata', {}).get('embedding')
    ])
    
    # Get embedding configuration for analysis
    embedding_config = settings.get_embedding_config()
    
    analysis = {
        'embedding_summary': {
            'total_tools': total_tools,
            'tools_with_embedding': tools_with_embedding,
            'embedding_coverage': round(tools_with_embedding / total_tools * 100, 2) if total_tools > 0 else 0,
            'embedding_model': embedding_config.get('model'),
            'embedding_dimensions': embedding_config.get('dimensions')
        },
        'processed_at': datetime.now().isoformat()
    }
    
    # Output result summary
    logger.info("=" * 60)
    logger.info("Embedding Calculation Result Summary")
    logger.info("=" * 60)
    logger.info(f"🔧 Total tools: {total_tools}")
    logger.info(f"✅ Tools with embedding: {tools_with_embedding}")
    logger.info(f"📊 Coverage rate: {analysis['embedding_summary']['embedding_coverage']}%")

    return analysis


def save_embedding_results(tools: List[Dict[str, Any]], logger):
    """Save embedding calculation results"""
    logger.info("Saving embedding calculation results...")
    
    try:
        tools_path = settings.get_data_path('tools')
        file_manager = FileManager(tools_path, logger)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save tool data with embeddings
        tools_file = f"tools_with_embeddings_{timestamp}.json"
        file_manager.save_json(tools, tools_file)
        logger.info(f"Saved tool data: {tools_file}")
        
        return {
            'tools_file': str(tools_path / tools_file),
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def main():
    """Main function"""
    print("🧮 Tool Embedding Calculator")
    print("=" * 50)
    
    # Setup logging
    logger = setup_embedding_logger()
    
    try:
        # Validate environment
        if not validate_environment(logger):
            return 1
        
        # Find latest tool file
        tools_file = find_latest_tools_file(logger)
        
        # Load tool data
        all_tools = load_tools_data(tools_file, logger)
        
        # Calculate embeddings
        tools_with_new_embeddings = compute_embeddings_for_tools(all_tools, logger)
        # Merge results
        all_updated_tools = merge_with_existing_tools(tools_with_new_embeddings, all_tools, logger)
        
        # Analyze results
        analysis = analyze_embedding_results(all_updated_tools, logger)
        
        # Save results
        save_info = save_embedding_results(all_updated_tools, logger)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 Embedding calculation completed!")
        logger.info(f"📁 Tool data: {save_info['tools_file']}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding calculation process failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)