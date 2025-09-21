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
Tool relationship graph construction script
Reads tool data containing embeddings, builds tool relationship graph and saves it
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
from modules.agent_synthesizer.tool_graph import ToolGraph


def setup_graph_logger():
    """Setup dedicated logger for tool graph construction"""
    logger = setup_logger(
        "tool_graph",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"],
        format_string=settings.LOGGING_CONFIG["format"]
    )
    return logger


def validate_environment(logger):
    """Validate runtime environment"""
    logger.info("Validating environment configuration...")
    
    # Check if tool data exists
    tools_path = settings.get_data_path('tools')
    if not tools_path.exists():
        logger.error(f"Tool data path does not exist: {tools_path}")
        return False
        
    return True


def find_latest_embedding_file(logger):
    """Find the latest embedding data file"""
    logger.info("Finding latest embedding data file...")
    
    tools_path = settings.get_data_path('tools')
    
    # Find embedding files
    embedding_files = list(tools_path.glob("tools_with_embeddings_*.json"))
    if embedding_files:
        latest_file = max(embedding_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using embedding file: {latest_file.name}")
        return latest_file
    
    raise FileNotFoundError("No tool data file containing embeddings found")


def load_tools_with_embeddings(file_path: Path, logger):
    """Load tool data containing embeddings"""
    logger.info(f"Loading tool data: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        # Validate embedding data
        tools_with_embedding = [
            tool for tool in tools 
            if tool.get('metadata', {}).get('embedding')
        ]
        
        logger.info(f"Successfully loaded {len(tools)} tools")
        
        if len(tools_with_embedding) == 0:
            raise ValueError("No tools containing embeddings found")
        
        return tools
        
    except Exception as e:
        logger.error(f"Failed to load tool data: {e}")
        raise


def build_tool_graph(tools: List[Dict[str, Any]], logger):
    """Build tool relationship graph"""
    logger.info("Starting to build tool relationship graph...")
    
    try:
        # Graph construction configuration
        graph_config = {
            'similarity_threshold': 0.7,          # Similarity threshold
            'min_similarity_threshold': 0.5,     # Minimum similarity threshold
            'max_edges_per_node': 10,             # Maximum edges per node
            'restart_probability': 0.15,          # Random walk restart probability
            'walk_length': 6                      # Walk length
        }
        
        input_data = {'tools': tools}
        
        with ToolGraph(graph_config, logger) as graph_module:
            graph_stats = graph_module.process(input_data)
            
        return graph_stats, graph_module
        
    except Exception as e:
        logger.error(f"Tool graph construction failed: {e}")
        raise


def test_graph_functionality(graph_module: ToolGraph, logger):
    """Test graph functionality"""
    logger.info("Testing graph functionality...")
    
    try:
        if graph_module.graph.number_of_nodes() == 0:
            logger.warning("No nodes in graph, skipping functionality test")
            return {}
        
        # Randomly select nodes for testing
        test_nodes = list(graph_module.graph.nodes())[:3]  # Test first 3 nodes
        
        test_results = []
        for node in test_nodes:
            # Test random walk
            related_tools = graph_module.random_walk_selection(node, count=5)
            
            # Test getting related tools
            direct_related = graph_module.get_related_tools(node, max_count=3)
            
            # Test tool clusters
            tool_cluster = graph_module.get_tool_cluster(node, max_size=6)
            
            tool_info = graph_module.tools_data.get(node, {})
            test_results.append({
                'tool_id': node,
                'tool_name': tool_info.get('name', 'Unknown'),
                'tool_category': tool_info.get('category', 'Unknown'),
                'random_walk_results': related_tools,
                'direct_related_count': direct_related,
                'cluster_size': len(tool_cluster)
            })
        
        logger.info("Graph functionality test results:")
        for result in test_results:
            logger.info(f"  Tool: {result['tool_name'][:30]} ({result['tool_category']})")
            logger.info(f"    Random walk selection: {result['random_walk_results']} related tools")
            logger.info(f"    Direct related: {result['direct_related_count']} tools")
            logger.info(f"    Cluster size: {result['cluster_size']} tools")
        
        return {
            'test_results': test_results,
            'total_tested_nodes': len(test_nodes)
        }
        
    except Exception as e:
        logger.error(f"Graph functionality test failed: {e}")
        return {}


def analyze_graph_quality(graph_stats: Dict[str, Any], graph_module: ToolGraph, logger):
    """Analyze graph quality"""
    logger.info("Analyzing graph quality...")
    
    try:
        # Basic graph statistics
        total_nodes = graph_stats.get('total_nodes', 0)
        total_edges = graph_stats.get('total_edges', 0)
        avg_degree = graph_stats.get('average_degree', 0)
        connected_components = graph_stats.get('connected_components', 0)
        largest_component = graph_stats.get('largest_component_size', 0)
        
        # Calculate graph quality metrics
        graph_density = (2 * total_edges) / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
        connectivity_ratio = largest_component / total_nodes if total_nodes > 0 else 0
        
        # Edge type distribution
        edge_types = graph_stats.get('edge_types', {})
        
        analysis = {
            'graph_quality': {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'average_degree': round(avg_degree, 2),
                'graph_density': round(graph_density, 4),
                'connected_components': connected_components,
                'largest_component_size': largest_component,
                'connectivity_ratio': round(connectivity_ratio, 4)
            },
            'edge_distribution': edge_types,
            'analyzed_at': datetime.now().isoformat()
        }
        
        # Output analysis results
        logger.info("=" * 60)
        logger.info("Tool Relationship Graph Quality Analysis")
        logger.info("=" * 60)
        logger.info(f"🔗 Number of nodes: {total_nodes}")
        logger.info(f"🔀 Number of edges: {total_edges}")
        logger.info(f"📊 Average degree: {avg_degree:.2f}")
        logger.info(f"🕸️ Graph density: {graph_density:.4f}")
        logger.info(f"🔗 Connected components: {connected_components}")
        logger.info(f"🏢 Largest connected component: {largest_component} ({connectivity_ratio:.1%})")
        
        logger.info("\n🔗 Edge type distribution:")
        for edge_type, count in edge_types.items():
            percentage = count / total_edges * 100 if total_edges > 0 else 0
            logger.info(f"   {edge_type}: {count} ({percentage:.1f}%)")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Graph quality analysis failed: {e}")
        return {}

def main():
    """Main function"""
    print("🕸️ Tool Relationship Graph Builder")
    print("=" * 50)
    
    # Setup logging
    logger = setup_graph_logger()
    
    try:
        # Validate environment
        if not validate_environment(logger):
            return 1
        
        # Find embedding file
        embedding_file = find_latest_embedding_file(logger)
        
        # Load tool data
        tools = load_tools_with_embeddings(embedding_file, logger)
        
        # Build tool graph
        graph_stats, graph_module = build_tool_graph(tools, logger)
        
        # Test graph functionality
        test_results = test_graph_functionality(graph_module, logger)
        
        # Analyze graph quality
        analysis = analyze_graph_quality(graph_stats, graph_module, logger)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎉 Tool relationship graph construction completed!")
        logger.info(f"📊 Analysis report: {analysis}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Tool graph construction process failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)