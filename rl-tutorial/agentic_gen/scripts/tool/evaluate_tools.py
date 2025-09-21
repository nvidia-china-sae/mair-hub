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
Tool Quality Evaluation Script
Batch evaluate tool quality using multi-threading
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
from modules.domain_tool_generator.tool_designer import ToolDesigner
from utils.logger import setup_logger
from utils.file_manager import FileManager


def find_latest_tools_file():
    """Find the latest tools file"""
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)

    batch_files = file_manager.list_files(".", "tools_batch_*.json")
    if batch_files:
        latest_file = max(batch_files, key=lambda f: file_manager.get_file_info(f)['modified'])
        return os.path.join(tools_dir, latest_file)
    
    return None


def load_tools_data(file_path: str):
    """Load tool data from specified file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    
    print(f"✅ Successfully loaded {len(tools_data)} tools")
    return tools_data



def validate_environment():
    """Validate environment configuration"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print(f"❌ Missing environment variable: OPENAI_API_KEY")
        print("Please ensure OPENAI_API_KEY is set in .env file")
        return False
    
    print("✅ Environment variable check passed")
    return True


def display_analysis_results(analysis: dict):
    """Display analysis results"""
    print("\n📈 Tool Quality Evaluation Results Analysis")
    print("="*60)
    
    print(f"📊 Basic Statistics:")
    print(f"  Total tool count: {analysis.get('total_count', 0)}")
    print(f"  Average quality score: {analysis.get('average_score', 0)}")
    print(f"  Score range: {analysis.get('min_score', 0)} - {analysis.get('max_score', 0)}")
    
    quality_summary = analysis.get('quality_summary', {})
    print(f"\n🎯 Quality Overview:")
    print(f"  High quality tool ratio: {quality_summary.get('high_quality_ratio', 0)}%")
    print(f"  Needs improvement ratio: {quality_summary.get('needs_improvement_ratio', 0)}%")
    
    score_dist = analysis.get('score_distribution', {})
    print(f"\n📊 Score Distribution:")
    print(f"  🌟 Excellent (≥4.5 points): {score_dist.get('excellent', 0)} tools")
    print(f"  ✅ Good (4.0-4.5 points): {score_dist.get('good', 0)} tools") 
    print(f"  ⚠️  Average (3.0-4.0 points): {score_dist.get('average', 0)} tools")
    print(f"  ❌ Poor (<3.0 points): {score_dist.get('poor', 0)} tools")
    
    recommendations = analysis.get('recommendations', {})
    if recommendations:
        print(f"\n💡 Recommendation Status Distribution:")
        for rec, count in recommendations.items():
            print(f"  {rec}: {count} tools")


def save_evaluation_results(evaluations: list, analysis: dict):
    """Save evaluation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tools_dir = settings.get_data_path('tools')
    file_manager = FileManager(tools_dir)
    
    # Save detailed evaluation results
    eval_filename = f"tool_evaluations_{timestamp}.json"
    file_manager.save_json(evaluations, eval_filename)
    print(f"💾 Detailed evaluation results saved: {eval_filename}")
    
    # Save analysis report
    analysis_filename = f"evaluation_analysis_{timestamp}.json"
    file_manager.save_json(analysis, analysis_filename)
    print(f"💾 Analysis report saved: {analysis_filename}")
    
    return eval_filename, analysis_filename


def main():
    """Main function"""
    print("🔍 Tool Quality Evaluation Started")
    print("="*60)
    
    # Validate environment
    if not validate_environment():
        return
    
    # Setup logging
    logger = setup_logger(
        "tool_evaluation",
        level=settings.LOGGING_CONFIG["level"],
        log_file=settings.LOGGING_CONFIG["file_path"]
    )
    
    try:
        # 1. Find latest tools file
        print("🔍 Finding latest tools file...")
        tools_file = find_latest_tools_file()
        if not tools_file:
            print("❌ No tools file found, please generate tools first")
            return
        
        print(f"📁 Using tools file: {os.path.basename(tools_file)}")
        
        # 2. Load tool data
        tools_data = load_tools_data(tools_file)
        if not tools_data:
            print("❌ Unable to load tool data, program exiting")
            return
        
        # 3. Initialize tool designer
        print("⚙️ Initializing tool designer...")
        
        # Get concurrency configuration
        concurrency_config = settings.CONCURRENCY_CONFIG
        max_workers = concurrency_config.get('max_workers', 4)
        
        designer_config = {
            'max_workers': max_workers,
        }
        
        tool_designer = ToolDesigner(designer_config, logger)
        tool_designer.initialize()
        
        print(f"🎯 Preparing to evaluate {len(tools_data)} tools")
        print(f"🔧 Using {tool_designer.max_workers} threads for parallel processing")
        
        # 4. Batch evaluate tool quality
        print("\n🔄 Starting batch tool quality evaluation...")
        start_time = datetime.now()
        
        evaluations = tool_designer.batch_evaluate_tools(tools_data)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️ Evaluation time: {processing_time:.2f} seconds")
        print(f"📈 Success rate: {len(evaluations)}/{len(tools_data)} ({len(evaluations)/len(tools_data)*100:.1f}%)")
        
        # 5. Analyze evaluation results
        analysis = tool_designer.analyze_evaluation_results(evaluations)
        display_analysis_results(analysis)
        
        # 6. Save results
        print(f"\n💾 Saving evaluation results...")
        eval_file, analysis_file = save_evaluation_results(evaluations, analysis)
        
        print(f"\n✅ Tool quality evaluation completed!")
        print(f"📁 Result files:")
        print(f"  - Detailed evaluation: {eval_file}")
        print(f"  - Analysis report: {analysis_file}")
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted execution")
    except Exception as e:
        logger.error(f"Tool evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()