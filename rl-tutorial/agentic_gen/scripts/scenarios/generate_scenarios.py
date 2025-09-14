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
Scenario Generation Script
Generate large amounts of scenario data based on all domains in configuration
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.logger import setup_logger
from modules.domain_tool_generator.scenario_generator import ScenarioGenerator


def setup_scenario_logger():
    """Setup scenario generation dedicated logger"""
    logger = setup_logger(
        "scenario_generation",
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
    
    logger.info("Environment validation completed")
    return True


def generate_scenarios_for_all_domains(logger):
    """Generate scenarios for all domains"""
    logger.info("Starting large-scale scenario generation...")
    
    # Get configuration
    scenario_config = settings.GENERATION_CONFIG['scenarios']
    domains = scenario_config['domains']
    target_total = scenario_config['target_count']
    
    logger.info(f"Target generation of {target_total} scenarios, covering {len(domains)} domains")
    
    try:
        # Initialize scenario generator
        generator_config = {
            'batch_size': scenario_config.get('batch_size', 10),
        }
        
        with ScenarioGenerator(generator_config, logger) as generator:
            # Prepare input data
            input_data = {
                'domains': domains,
                'target_count': target_total
            }
            
            # Execute generation
            logger.info("Generating scenarios...")
            scenarios = generator.process(input_data)
            
            # Get generation statistics
            stats = generator.get_generation_stats()
            
            return scenarios, stats
            
    except Exception as e:
        logger.error(f"Scenario generation failed: {e}")
        raise


def analyze_generation_results(scenarios: List[Dict[str, Any]], stats: Dict[str, Any], logger):
    """Analyze generation results"""
    logger.info("\n" + "="*50)
    logger.info("Scenario Generation Results Analysis")
    logger.info("="*50)
    
    # Basic statistics
    total_scenarios = len(scenarios)
    logger.info(f"📊 Overall Statistics:")
    logger.info(f"   Total scenarios generated: {total_scenarios}")
    logger.info(f"   Generation batches: {stats.get('batch_files', 0)}")
    
    # Domain distribution
    domain_distribution = {}
    for scenario in scenarios:
        domain = scenario.get('domain', 'Unknown')
        domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
    
    logger.info(f"\n🌐 Domain Distribution:")
    for domain, count in sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_scenarios) * 100 if total_scenarios > 0 else 0
        logger.info(f"   {domain}: {count} scenarios ({percentage:.1f}%)")
    
    logger.info("\n" + "="*50)


def main():
    """Main function"""
    try:
        # Setup logging
        logger = setup_scenario_logger()
        
        logger.info("🚀 Starting large-scale scenario generation script")
        logger.info("="*60)
        
        # Validate environment
        if not validate_environment(logger):
            sys.exit(1)
        
        # Generate scenarios
        scenarios, stats = generate_scenarios_for_all_domains(logger)
        
        # Analyze results
        analyze_generation_results(scenarios, stats, logger)
        
        # Final summary
        logger.info(f"\n🎉 Scenario generation completed!")
        logger.info(f"✅ Successfully generated {len(scenarios)} scenarios")

        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted program execution")
        return 0
    except Exception as e:
        print(f"❌ Program execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 