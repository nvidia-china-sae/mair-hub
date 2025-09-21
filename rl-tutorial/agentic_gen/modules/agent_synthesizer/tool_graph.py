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
Tool relationship graph construction and random walk module
Used to build relationship graphs between tools and select related tools through random walks
"""

import random
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from datetime import datetime

import networkx as nx
import numpy as np
from core.base_module import BaseModule
from core.exceptions import AgentDataGenException
from utils.file_manager import FileManager
from utils.data_processor import DataProcessor


class ToolGraph(BaseModule):
    """Tool relationship graph builder"""
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize tool relationship graph builder
        
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        super().__init__(config, logger)
        
        self.graph = None
        self.tools_data = {}  # tool_id -> tool_data mapping
        self.file_manager = None
        self.data_processor = None
        
        # Similarity threshold configuration
        self.similarity_threshold = 0.7
        self.min_similarity_threshold = 0.5
        self.max_edges_per_node = 10
        
        # Random walk parameters
        self.walk_length = 6  # Maximum walk length
        self.restart_probability = 0.15  # Restart probability
        
    def _setup(self):
        """Setup components"""
        from config.settings import settings
        
        # Initialize file manager
        data_path = settings.get_data_path('tools')
        self.file_manager = FileManager(data_path, self.logger)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.logger)
        
        # Initialize graph
        self.graph = nx.Graph()
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Build tool relationship graph
        
        Args:
            input_data: Dictionary containing tool data
            **kwargs: Other parameters
            
        Returns:
            Dictionary containing graph statistics
        """
        try:
            tools = input_data.get('tools', [])
            
            if not tools:
                raise AgentDataGenException("No tools data provided")
            
            # Build graph
            self.build_graph(tools)
            
            # Generate statistics
            stats = self._generate_graph_stats()
            
            # Save graph data
            self._save_graph_data()
            
            self.logger.info(f"Successfully built tool graph with {len(tools)} tools")
            return stats
            
        except Exception as e:
            self.logger.error(f"Tool graph building failed: {e}")
            raise AgentDataGenException(f"Failed to build tool graph: {e}")
    
    def build_graph(self, tools: List[Dict[str, Any]]) -> None:
        """
        Build tool relationship graph
        
        Args:
            tools: Tool list
        """
        try:
            self.logger.info(f"Building tool graph for {len(tools)} tools")
            
            # Clear existing graph
            self.graph.clear()
            self.tools_data.clear()
            
            # Add nodes
            for tool in tools:
                tool_id = tool.get('id')
                if tool_id:
                    self.graph.add_node(tool_id, **tool)
                    self.tools_data[tool_id] = tool
            
            # Build edges (based on similarity)
            self._build_edges_by_similarity(tools)
            
            
            self.logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}")
            raise AgentDataGenException(f"Graph building failed: {e}")
    
    def _build_edges_by_similarity(self, tools: List[Dict[str, Any]]):
        """Build edges based on embedding similarity"""
        tools_with_embeddings = [
            tool for tool in tools 
            if tool.get('metadata').get('embedding')
        ]
        
        self.logger.info(f"Building similarity edges for {len(tools_with_embeddings)} tools with embeddings")
        
        # Calculate similarity between all tool pairs
        for i, tool1 in enumerate(tools_with_embeddings):
            tool1_id = tool1.get('id')
            tool1_embedding = tool1['metadata']['embedding']
            
            similarities = []
            for j, tool2 in enumerate(tools_with_embeddings):
                if i >= j:  # Avoid duplicate calculations
                    continue
                    
                tool2_id = tool2.get('id')
                tool2_embedding = tool2['metadata']['embedding']
                
                similarity = self._calculate_cosine_similarity(tool1_embedding, tool2_embedding)
                
                if similarity >= self.min_similarity_threshold:
                    similarities.append((tool2_id, similarity))
            
            # Keep most similar tools as neighbors for each tool
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for tool2_id, similarity in similarities[:self.max_edges_per_node]:
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(tool1_id, tool2_id, weight=similarity, edge_type='similarity')
    
    def _build_edges_by_category_and_domain(self, tools: List[Dict[str, Any]]):
        """Build edges based on category and domain"""
        category_groups = {}
        domain_groups = {}
        
        # Group by category and domain
        for tool in tools:
            tool_id = tool.get('id')
            category = tool.get('category', '')
            domain = tool.get('metadata', {}).get('domain', '')
            
            if category:
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(tool_id)
            
            if domain:
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(tool_id)
        
        # Add edges between tools of same category
        for category, tool_ids in category_groups.items():
            if len(tool_ids) > 1:
                self._add_group_edges(tool_ids, 'category', weight=0.6)
        
        # Add edges between tools of same domain
        for domain, tool_ids in domain_groups.items():
            if len(tool_ids) > 1:
                self._add_group_edges(tool_ids, 'domain', weight=0.5)
    
    def _add_group_edges(self, tool_ids: List[str], edge_type: str, weight: float):
        """Add edges for tool group"""
        # Avoid complete connection, only connect some tools
        for i, tool1_id in enumerate(tool_ids):
            # Randomly select several tools to connect
            connection_count = min(3, len(tool_ids) - 1)
            other_tools = [tid for tid in tool_ids if tid != tool1_id]
            connected_tools = random.sample(other_tools, min(connection_count, len(other_tools)))
            
            for tool2_id in connected_tools:
                if not self.graph.has_edge(tool1_id, tool2_id):
                    self.graph.add_edge(tool1_id, tool2_id, weight=weight, edge_type=edge_type)
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def random_walk_selection(self, start_tool: str, count: int, restart_prob: float = None) -> List[str]:
        """
        Select related tools using random walk
        
        Args:
            start_tool: Starting tool ID
            count: Number of tools to select
            restart_prob: Restart probability
            
        Returns:
            List of selected tool IDs
        """
        if start_tool not in self.graph:
            self.logger.warning(f"Start tool {start_tool} not found in graph")
            return []
        
        restart_prob = restart_prob or self.restart_probability
        selected_tools = set()
        current_tool = start_tool
        
        # Random walk
        for _ in range(self.walk_length * count):
            if len(selected_tools) >= count:
                break
            
            # Add current tool
            selected_tools.add(current_tool)
            
            # Decide whether to restart
            if random.random() < restart_prob:
                current_tool = start_tool
                continue
            
            # Get neighbor nodes
            neighbors = list(self.graph.neighbors(current_tool))
            if not neighbors:
                current_tool = start_tool
                continue
            
            # Select next node based on edge weights
            next_tool = self._weighted_random_choice(current_tool, neighbors)
            current_tool = next_tool
        
        # Remove starting tool as it's not counted in selection results
        selected_tools.discard(start_tool)
        
        # If insufficient tools selected, supplement with most similar tools
        if len(selected_tools) < count:
            additional_tools = self.get_related_tools(start_tool, count - len(selected_tools))
            selected_tools.update(additional_tools)
        
        return list(selected_tools)[:count]
    
    def _weighted_random_choice(self, current_tool: str, neighbors: List[str]) -> str:
        """Randomly select neighbor based on edge weights"""
        try:
            weights = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(current_tool, neighbor)
                weight = edge_data.get('weight', 0.5) if edge_data else 0.5
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                return random.choice(neighbors)
            
            weights = [w / total_weight for w in weights]
            
            # Randomly select based on weights
            return np.random.choice(neighbors, p=weights)
            
        except Exception as e:
            self.logger.error(f"Failed to make weighted random choice: {e}")
            return random.choice(neighbors)
    
    def get_related_tools(self, tool_id: str, max_count: int) -> List[str]:
        """
        Get tools related to specified tool
        
        Args:
            tool_id: Tool ID
            max_count: Maximum count
            
        Returns:
            List of related tool IDs
        """
        if tool_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(tool_id))
        
        neighbor_weights = []
        for neighbor in neighbors:
            edge_data = self.graph.get_edge_data(tool_id, neighbor)
            weight = edge_data.get('weight', 0.5) if edge_data else 0.5
            neighbor_weights.append((neighbor, weight))
        
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Return first max_count neighbors
        return [neighbor for neighbor, _ in neighbor_weights[:max_count]]
    
    def get_tool_cluster(self, tool_id: str, max_size: int = 6) -> List[str]:
        """
        Get tool cluster containing specified tool
        
        Args:
            tool_id: Tool ID
            max_size: Maximum cluster size
            
        Returns:
            List of tool IDs in the cluster
        """
        if tool_id not in self.graph:
            return [tool_id] if tool_id in self.tools_data else []
        
        # Use BFS to build tool cluster
        visited = set()
        queue = [tool_id]
        cluster = []
        
        while queue and len(cluster) < max_size:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            cluster.append(current)
            
            # Add neighbors to queue
            neighbors = list(self.graph.neighbors(current))
            for neighbor in neighbors:
                if neighbor not in visited and len(cluster) + len(queue) < max_size:
                    queue.append(neighbor)
        
        return cluster
    
    def _load_tools_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load tools data"""
        try:
            return self.file_manager.load_json(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load tools data: {e}")
            return []
    
    def _generate_graph_stats(self) -> Dict[str, Any]:
        """Generate graph statistics"""
        try:
            stats = {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
                'connected_components': nx.number_connected_components(self.graph),
                'largest_component_size': len(max(nx.connected_components(self.graph), key=len)) if self.graph.number_of_nodes() > 0 else 0,
                'edge_types': self._count_edge_types(),
                'generated_at': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to generate graph stats: {e}")
            return {}
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edge types"""
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        return edge_types
    
    def _save_graph_data(self):
        """Save graph data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save graph in edge list format
            edges_data = []
            for u, v, data in self.graph.edges(data=True):
                edges_data.append({
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 0.5),
                    'edge_type': data.get('edge_type', 'unknown')
                })
            
            graph_data = {
                'nodes': list(self.graph.nodes()),
                'edges': edges_data,
                'stats': self._generate_graph_stats()
            }
            
            filename = f"tool_graph_{timestamp}.json"
            self.file_manager.save_json(graph_data, filename)
            
            self.logger.info(f"Saved tool graph to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save graph data: {e}")
    
    def load_graph_from_file(self, file_path: str) -> bool:
        """
        Load graph data from file
        
        Args:
            file_path: Graph data file path
            
        Returns:
            Whether loading was successful
        """
        try:
            graph_data = self.file_manager.load_json(file_path)
            
            self.graph.clear()
            
            for node_id in graph_data.get('nodes', []):
                if node_id in self.tools_data:
                    self.graph.add_node(node_id, **self.tools_data[node_id])
                else:
                    self.graph.add_node(node_id)
            
            for edge in graph_data.get('edges', []):
                self.graph.add_edge(
                    edge['source'], 
                    edge['target'],
                    weight=edge.get('weight', 0.5),
                    edge_type=edge.get('edge_type', 'unknown')
                )
            
            self.logger.info(f"Successfully loaded graph from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load graph from file: {e}")
            return False