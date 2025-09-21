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
Tool Filtering Script
Filter low-quality tools based on quality evaluation results and deduplicate based on embedding similarity
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from utils.file_manager import FileManager


def setup_filter_logger():
    """Setup tool filtering dedicated logger"""
    logger = setup_logger(
        "tool_filter",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    return logger


def validate_environment():
    """Validate environment configuration"""
    required_files = []
    
    # Check required environment variables
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('DASHSCOPE_API_KEY'):
        print("❌ Missing API key, need to set OPENAI_API_KEY or DASHSCOPE_API_KEY")
        return False
    
    print("✅ Environment variable check passed")
    return True


def find_latest_files():
    """Find latest tool and evaluation files"""
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Find tool files (prioritize files with embeddings)
    embedding_files = file_manager.list_files(".", "*tools_with_embeddings*.json")
    tools_file = None
    if embedding_files:
        tools_file = max(embedding_files, key=lambda f: file_manager.get_file_info(f)['modified'])
    
    # Find evaluation files
    evaluation_files = file_manager.list_files(".", "*tool_evaluations*.json")
    evaluation_file = None
    if evaluation_files:
        evaluation_file = max(evaluation_files, key=lambda f: file_manager.get_file_info(f)['modified'])
    
    return tools_file, evaluation_file


def load_data_files(tools_file: str, evaluation_file: str):
    """Load tool and evaluation data files"""
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Load tool data
    if not tools_file:
        raise FileNotFoundError("Tool data file not found")
    
    print(f"📂 Loading tool data: {os.path.basename(tools_file)}")
    tools_data = file_manager.load_json(os.path.basename(tools_file))
    print(f"✅ Successfully loaded {len(tools_data)} tools")
    
    # Load evaluation data
    evaluations_data = []
    if evaluation_file:
        print(f"📂 Loading evaluation data: {os.path.basename(evaluation_file)}")
        evaluations_data = file_manager.load_json(os.path.basename(evaluation_file))
        print(f"✅ Successfully loaded {len(evaluations_data)} evaluation results")
    else:
        print("⚠️  Evaluation file not found, will skip quality filtering step")
    
    return tools_data, evaluations_data


def filter_tools_by_quality(tools_data: List[Dict], evaluations_data: List[Dict], 
                           quality_threshold: float = 4.0) -> Tuple[List[Dict], Dict]:
    """Filter tools based on quality evaluation results"""
    if not evaluations_data:
        print("📊 Skipping quality filtering step")
        return tools_data, {'skipped': True}
    
    print(f"🔍 Filtering tools based on quality threshold {quality_threshold}...")
    
    # Create evaluation result mapping
    evaluation_map = {}
    for eval_item in evaluations_data:
        tool_id = eval_item.get('tool_id')
        total_score = eval_item.get('overall_score', 0)
        if tool_id:
            evaluation_map[tool_id] = total_score
    
    # Filter high-quality tools
    high_quality_tools = []
    quality_stats = {
        'total_tools': len(tools_data),
        'evaluated_tools': 0,
        'high_quality_tools': 0,
        'filtered_out': 0,
        'no_evaluation': 0
    }
    
    for tool in tools_data:
        tool_id = tool.get('id')
        
        if tool_id in evaluation_map:
            quality_stats['evaluated_tools'] += 1
            score = evaluation_map[tool_id]
            
            if score >= quality_threshold:
                high_quality_tools.append(tool)
                quality_stats['high_quality_tools'] += 1
            else:
                quality_stats['filtered_out'] += 1
        else:
            # Tools without evaluation are kept by default
            high_quality_tools.append(tool)
            quality_stats['no_evaluation'] += 1
    
    print(f"📈 Quality filtering results:")
    print(f"  Total tools: {quality_stats['total_tools']}")
    print(f"  Evaluated tools: {quality_stats['evaluated_tools']}")
    print(f"  High-quality tools: {quality_stats['high_quality_tools']}")
    print(f"  Filtered out tools: {quality_stats['filtered_out']}")
    print(f"  Unevaluated tools: {quality_stats['no_evaluation']}")
    
    return high_quality_tools, quality_stats


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embedding vectors"""
    try:
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            return 0.0
        
        # Manually calculate cosine similarity to avoid sklearn dependency
        import math
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate vector norms
        norm_a = math.sqrt(sum(a * a for a in embedding1))
        norm_b = math.sqrt(sum(b * b for b in embedding2))
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (norm_a * norm_b)
        return float(max(0.0, min(1.0, similarity)))  # Ensure result is in [0,1] range
        
    except Exception:
        return 0.0


def group_tools_by_scenario(tools_data: List[Dict]) -> Dict[str, List[Dict]]:
    """Group tools by scenario"""
    scenario_groups = defaultdict(list)
    
    for tool in tools_data:
        scenario_ids = tool.get('scenario_ids', [])
        
        if scenario_ids:
            # Use first scenario as primary scenario
            primary_scenario = scenario_ids[0]
            scenario_groups[primary_scenario].append(tool)
        else:
            # Tools without scenarios are grouped separately
            scenario_groups['no_scenario'].append(tool)
    
    return scenario_groups


def deduplicate_tools_in_scenario(tools_in_scenario: List[Dict], 
                                similarity_threshold: float = 0.8) -> Tuple[List[Dict], Dict]:
    """Deduplicate tools within scenario based on embedding similarity"""
    if len(tools_in_scenario) <= 1:
        return tools_in_scenario, {'clusters': 0, 'removed': 0}
    
    # Filter tools with embeddings
    tools_with_embedding = []
    tools_without_embedding = []
    
    for tool in tools_in_scenario:
        embedding = tool.get('metadata', {}).get('embedding')
        if embedding and any(x != 0.0 for x in embedding):
            tools_with_embedding.append(tool)
        else:
            tools_without_embedding.append(tool)
    
    # If no embeddings, return directly
    if len(tools_with_embedding) <= 1:
        return tools_in_scenario, {'clusters': 0, 'removed': 0}
    
    # Calculate similarity matrix and cluster
    clusters = []
    used_indices = set()
    
    for i, tool1 in enumerate(tools_with_embedding):
        if i in used_indices:
            continue
        
        # Create new cluster
        cluster = [i]
        embedding1 = tool1['metadata']['embedding']
        
        # Find similar tools
        for j, tool2 in enumerate(tools_with_embedding[i+1:], i+1):
            if j in used_indices:
                continue
            
            embedding2 = tool2['metadata']['embedding']
            similarity = calculate_cosine_similarity(embedding1, embedding2)
            
            if similarity >= similarity_threshold:
                cluster.append(j)
                used_indices.add(j)
        
        clusters.append(cluster)
        used_indices.add(i)
    
    # Select best tool from each cluster
    selected_tools = []
    removed_count = 0
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Keep standalone tools directly
            selected_tools.append(tools_with_embedding[cluster[0]])
        else:
            # Select best tool from cluster (here selecting first, can be optimized with other criteria)
            best_tool = tools_with_embedding[cluster[0]]
            selected_tools.append(best_tool)
            removed_count += len(cluster) - 1
    
    # Add tools without embeddings
    selected_tools.extend(tools_without_embedding)
    
    dedup_stats = {
        'clusters': len(clusters),
        'removed': removed_count,
        'original_count': len(tools_in_scenario),
        'final_count': len(selected_tools)
    }
    
    return selected_tools, dedup_stats


def filter_duplicate_tools(tools_data: List[Dict], similarity_threshold: float = 0.85) -> Tuple[List[Dict], Dict]:
    """Deduplicate within each scenario based on embedding similarity"""
    print(f"🔄 Deduplicating based on embedding similarity (threshold: {similarity_threshold})...")
    
    # Group by scenario
    scenario_groups = group_tools_by_scenario(tools_data)
    print(f"📊 Found {len(scenario_groups)} scenario groups")
    
    # Deduplicate within each scenario
    final_tools = []
    total_stats = {
        'total_scenarios': len(scenario_groups),
        'total_clusters': 0,
        'total_removed': 0,
        'original_total': len(tools_data),
        'scenario_details': {}
    }
    
    for scenario_id, tools_in_scenario in scenario_groups.items():
        if len(tools_in_scenario) > 1:
            deduplicated_tools, dedup_stats = deduplicate_tools_in_scenario(
                tools_in_scenario, similarity_threshold
            )
            
            final_tools.extend(deduplicated_tools)
            total_stats['total_clusters'] += dedup_stats['clusters']
            total_stats['total_removed'] += dedup_stats['removed']
            total_stats['scenario_details'][scenario_id] = dedup_stats
            
            if dedup_stats['removed'] > 0:
                print(f"  Scenario {scenario_id}: {dedup_stats['original_count']} → {dedup_stats['final_count']} "
                      f"(-{dedup_stats['removed']})")
        else:
            final_tools.extend(tools_in_scenario)
            total_stats['scenario_details'][scenario_id] = {
                'clusters': 0, 'removed': 0, 
                'original_count': len(tools_in_scenario),
                'final_count': len(tools_in_scenario)
            }
    
    total_stats['final_total'] = len(final_tools)
    
    print(f"📈 Deduplication results:")
    print(f"  Original tool count: {total_stats['original_total']}")
    print(f"  Final tool count: {total_stats['final_total']}")
    print(f"  Removed tool count: {total_stats['total_removed']}")
    
    return final_tools, total_stats


def save_filtered_tools(tools_data: List[Dict], quality_stats: Dict, dedup_stats: Dict):
    """Save filtered tools"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Save final tool data
    final_tools_file = f"final_tools_{timestamp}.json"
    file_manager.save_json(tools_data, final_tools_file)
    print(f"💾 Final tool data saved: {final_tools_file}")
    
    # Save filtering statistics report
    filter_report = {
        'filter_summary': {
            'timestamp': timestamp,
            'final_tool_count': len(tools_data),
            'quality_filter_applied': not quality_stats.get('skipped', False),
            'similarity_deduplication_applied': True
        },
        'quality_filter_stats': quality_stats,
        'deduplication_stats': dedup_stats,
        'process_metadata': {
            'similarity_threshold': 0.8,
            'quality_threshold': 4.0,
            'processed_at': datetime.now().isoformat()
        }
    }
    
    report_file = f"filter_report_{timestamp}.json"
    file_manager.save_json(filter_report, report_file)
    print(f"💾 Filtering report saved: {report_file}")
    
    return final_tools_file, report_file


def main():
    """Main function"""
    print("🔧 Tool Filter")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
    # Setup logging
    logger = setup_filter_logger()
    
    try:
        # 1. Find latest files
        print("🔍 Finding latest tool and evaluation files...")
        tools_file, evaluation_file = find_latest_files()
        
        if not tools_file:
            print("❌ Tool data file not found")
            return
        
        # 2. Load data
        tools_data, evaluations_data = load_data_files(tools_file, evaluation_file)
        
        # 3. Quality filtering
        filtered_tools, quality_stats = filter_tools_by_quality(
            tools_data, evaluations_data, quality_threshold=4.0
        )
        
        # 4. Similarity deduplication
        final_tools, dedup_stats = filter_duplicate_tools(
            filtered_tools, similarity_threshold=0.85
        )
        
        # 5. Save results
        print(f"\n💾 Saving filtering results...")
        final_file, report_file = save_filtered_tools(final_tools, quality_stats, dedup_stats)
        
        print(f"\n✅ Tool filtering completed!")
        print(f"📁 Result files:")
        print(f"  - Final tools: {final_file}")
        print(f"  - Filtering report: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
    except Exception as e:
        logger.error(f"Tool filtering failed: {e}")
        print(f"❌ Filtering failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
