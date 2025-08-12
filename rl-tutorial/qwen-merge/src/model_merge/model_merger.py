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
Model merger tool - Merge VLM models with LLM models for architectural compatibility

Main functions:
1. Analyze parameter structure differences between two models
2. Remove parameters from VLM that don't exist in LLM
3. Add parameters from LLM that don't exist in VLM
4. Initialize visual projection layers
5. Save the merged model
6. Copy files from materials directory
"""

import torch
import json
import re
import os
import shutil
import argparse
from typing import Dict, Set, Tuple
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoModelForCausalLM
)
from huggingface_hub import snapshot_download


class ModelMerger:
    """Model merger class"""
    
    def __init__(self, vlm_path: str, llm_path: str, materials_path: str = "./materials"):
        """
        Initialize model merger
        
        Args:
            vlm_path: VLM model path or Hugging Face model ID
            llm_path: LLM model path or Hugging Face model ID
            materials_path: Materials folder path
        """
        # Parse and get actual local paths
        self.vlm_path = self._resolve_model_path(vlm_path)
        self.llm_path = self._resolve_model_path(llm_path)
        self.materials_path = materials_path
        self.vlm_config = None
        self.llm_config = None
        self.vlm = None  # Keep vlm instance for saving
        
        # Load config files
        self.vlm_config = json.load(open(os.path.join(self.vlm_path, "config.json")))
        self.llm_config = json.load(open(os.path.join(self.llm_path, "config.json")))
        print("Config files loaded successfully")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolve model path, download to cache if it's a HF model ID and return local path
        
        Args:
            model_path: Model path or Hugging Face model ID
            
        Returns:
            Resolved local path
        """
        # Check if it's a local path
        if os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Using local model: {model_path}")
            return model_path
        else:
            # Assume it's a HF model ID, download to cache
            print(f"Local path doesn't exist, downloading from Hugging Face: {model_path}")
            try:
                local_path = snapshot_download(model_path)
                print(f"Model downloaded to cache: {local_path}")
                return local_path
            except Exception as e:
                print(f"Failed to download model: {e}")
                raise ValueError(f"Unable to load model {model_path}: local path doesn't exist and HF download failed")
    
    def _convert_vlm_to_llm_param_name(self, vlm_param_name: str) -> str:
        """
        Convert VLM parameter name to corresponding LLM parameter name format
        
        Args:
            vlm_param_name: VLM parameter name
            
        Returns:
            Converted LLM format parameter name
        """
        # Handle VLM parameter name changes after loading:
        # - model.language_model.xxx -> model.xxx (LLM format)
        # - model.visual.xxx -> keep unchanged (visual parameters)
        # - lm_head.weight -> keep unchanged
        
        if vlm_param_name.startswith("model.language_model."):
            return vlm_param_name.replace("model.language_model.", "model.")
        elif vlm_param_name.startswith("model.visual."):
            return vlm_param_name.replace("model.visual.", "visual.")
        else:
            return vlm_param_name
    
    def _normalize_parameter_name(self, name: str) -> str:
        """
        Unified parameter name normalization function - for comparing parameter structures
        
        Args:
            name: Original parameter name
            
        Returns:
            Normalized parameter name (removing layer numbers and prefix differences)
        """
        # First convert VLM parameter name format
        name = self._convert_vlm_to_llm_param_name(name)
        
        # Normalization: remove numeric layer numbers for structure comparison
        if "visual" in name and "blocks." in name:
            # For visual.blocks.number.xxx parameters, remove layer numbers
            return re.sub(r'\.blocks\.\d+\.', '.blocks..', name)
        elif "layers." in name:
            # For model.layers.number.xxx parameters, remove layer numbers
            return re.sub(r'\.layers\.\d+\.', '.layers..', name)
        else:
            # Return other parameters directly
            return name
    
    def _format_set_for_display(self, param_set: Set[str], title: str, max_display: int = 10) -> None:
        """
        Format and display parameter sets, with normalization for observation
        
        Args:
            param_set: Parameter name set
            title: Display title
            max_display: Maximum display count
        """
        if not param_set:
            print(f"{title}: None")
            return
            
        # Group parameters by pattern
        pattern_groups = {}
        for param in param_set:
            # Further normalize, replace numbers with placeholders
            pattern = re.sub(r'\d+', 'N', param)
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(param)
        
        print(f"{title} ({len(param_set)} items):")
        
        # Display by pattern
        for i, (pattern, params) in enumerate(sorted(pattern_groups.items()), 1):
            if len(params) == 1:
                print(f"  {i:2d}. {params[0]}")
            else:
                print(f"  {i:2d}. {pattern} (total {len(params)} items)")
                if len(params) <= 3:  # If count is small, show specific ones
                    for param in sorted(params):
                        print(f"      - {param}")
            
            # Limit display count
            if i >= max_display:
                remaining = len(pattern_groups) - max_display
                if remaining > 0:
                    print(f"  ... {remaining} more patterns not displayed")
                break

    def _analyze_parameter_differences(self, vlm_state_dict: Dict, llm_state_dict: Dict) -> Tuple[Set[str], Set[str], Set[str], Set[str], Dict[str, str]]:
        """
        Analyze parameter structure differences between two models
        
        Args:
            vlm_state_dict: VLM model parameter dictionary
            llm_state_dict: LLM model parameter dictionary
            
        Returns:
            Tuple[replace_set, del_set, add_set, visual_keep_set, vlm_param_mapping]: 
            Parameters to replace, parameters to delete, parameters to add, Visual parameters to always keep, VLM parameter mapping dictionary
        """
        # Extract LLM parameter names - using unified normalization function
        llm_name_set = set()
        for name in llm_state_dict.keys():
            normalized_name = self._normalize_parameter_name(name)
            llm_name_set.add(normalized_name)
        
        # Extract VLM parameter names - using unified normalization function and create mapping dictionary
        vlm_name_set = set()
        vlm_visual_params_normalized = set()
        vlm_param_mapping = {}  # Normalized name -> original parameter name list mapping
        
        for name in vlm_state_dict.keys():
            # First convert parameter name format
            converted_name = self._convert_vlm_to_llm_param_name(name)
            normalized_name = self._normalize_parameter_name(name)
            
            # Establish mapping relationship
            if normalized_name not in vlm_param_mapping:
                vlm_param_mapping[normalized_name] = []
            vlm_param_mapping[normalized_name].append(name)
            
            if "visual" in converted_name:
                vlm_visual_params_normalized.add(normalized_name)
            else:
                vlm_name_set.add(normalized_name)
        
        # Analyze differences
        replace_set = llm_name_set & vlm_name_set
        del_set = vlm_name_set - llm_name_set
        add_set = llm_name_set - vlm_name_set
        visual_keep_set = vlm_visual_params_normalized
        
        # Display results using new formatting function
        self._format_set_for_display(replace_set, "Parameter types updated from LLM")
        self._format_set_for_display(del_set, "Parameter types to be deleted")
        self._format_set_for_display(add_set, "Parameter types to be added")
        self._format_set_for_display(visual_keep_set, "Visual parameter types (always keep)")
        
        # Print mapping info (for debugging only)
        print(f"\nVLM parameter mapping statistics:")
        for normalized_name, original_names in vlm_param_mapping.items():
            if len(original_names) > 1:  # Only show one-to-many mappings
                print(f"  {normalized_name} -> {len(original_names)} parameters")
        
        return replace_set, del_set, add_set, visual_keep_set, vlm_param_mapping
    
    def _initialize_visual_projection(self, merged_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Initialize visual projection layer
        
        Args:
            merged_state_dict: Merged state_dict
        """
        vlm_output = self.vlm_config["vision_config"]["out_hidden_size"]
        llm_input = self.llm_config["hidden_size"]
        
        print(f"Initializing visual projection layer: {vlm_output} -> {llm_input}")
        
        # Initialize linear layer on CPU
        linear = torch.nn.Linear(vlm_output, llm_input, bias=True)
        linear.weight.data.normal_(mean=0.0, std=0.02)
        linear.bias.data.zero_()
        
        # Add to state_dict (ensure on CPU and memory contiguous)
        merged_state_dict["visual.merger.mlp.4.weight"] = linear.weight.detach().contiguous()
        merged_state_dict["visual.merger.mlp.4.bias"] = linear.bias.detach().contiguous()
        
        print("Visual projection layer initialization completed")
    
    def merge_models(self) -> Dict[str, torch.Tensor]:
        """
        Execute model merging
        
        Returns:
            Merged state_dict
        """
        print("Starting model merging...")
        
        # Load VLM model on GPU
        print(f"Loading VLM model: {self.vlm_path}")
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_path, 
            device_map="auto"
        )
        vlm_state_dict = self.vlm.state_dict()
        
        # Debug: print some VLM parameter names
        print("\n=== VLM Parameter Names Debug ===")
        visual_params = [name for name in vlm_state_dict.keys() if "visual" in name]
        language_params = [name for name in vlm_state_dict.keys() if "language_model" in name]
        other_params = [name for name in vlm_state_dict.keys() if "visual" not in name and "language_model" not in name]
        
        print(f"Total parameters containing 'visual' in VLM: {len(visual_params)}")
        print(f"Total parameters containing 'language_model' in VLM: {len(language_params)}")
        print(f"Total other parameters in VLM: {len(other_params)}")
        
        print("First 5 visual parameter names:")
        for i, name in enumerate(visual_params[:5]):
            print(f"  {i+1:2d}. {name}")
            
        print("First 5 language_model parameter names:")
        for i, name in enumerate(language_params[:5]):
            print(f"  {i+1:2d}. {name}")
            
        print("First 5 other parameter names:")
        for i, name in enumerate(other_params[:5]):
            print(f"  {i+1:2d}. {name}")
        
        # Load LLM model on GPU
        print(f"\nLoading LLM model: {self.llm_path}")
        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path, 
            device_map="auto"
        )
        llm_state_dict = llm.state_dict()
        
        # Debug: print some LLM parameter names
        print("\n=== LLM Parameter Names Debug ===")
        llm_params = list(llm_state_dict.keys())
        print(f"Total parameters in LLM: {len(llm_params)}")
        print("First 10 LLM parameter names:")
        for i, name in enumerate(llm_params[:10]):
            print(f"  {i+1:2d}. {name}")
        if len(llm_params) > 10:
            print(f"  ... {len(llm_params) - 10} more LLM parameters")
        
        # Analyze parameter differences (now returns mapping dictionary)
        replace_set, del_set, add_set, visual_keep_set, vlm_param_mapping = self._analyze_parameter_differences(vlm_state_dict, llm_state_dict)
        
        # Create new merged dictionary
        merged_state_dict = {}
        
        # Check tie_word_embeddings setting in LLM config
        tie_word_embeddings = self.llm_config.get("tie_word_embeddings", False)
        print(f"LLM tie_word_embeddings setting: {tie_word_embeddings}")
        
        # 1. Add replace_set and add_set parameters from LLM
        llm_added_count = 0
        llm_added_names = []  # Collect added parameter names
        skipped_lm_head = False  # Record if lm_head.weight was skipped
        
        for name, param in llm_state_dict.items():
            normalized_name = self._normalize_parameter_name(name)
            
            if normalized_name in replace_set or normalized_name in add_set:
                # Special handling for lm_head.weight parameter
                if name == "lm_head.weight":
                    if tie_word_embeddings:
                        print(f"  Skipping lm_head.weight (tie_word_embeddings=True)")
                        skipped_lm_head = True
                        continue
                    else:
                        print(f"  Adding lm_head.weight (tie_word_embeddings=False)")
                
                merged_state_dict[name] = param.detach().clone().contiguous()
                llm_added_count += 1
                llm_added_names.append(name)  # Record parameter name
        
        print(f"Added {llm_added_count} parameters from LLM:")
        for i, name in enumerate(llm_added_names, 1):
            print(f"  {i:3d}. {name}")
        
        if skipped_lm_head:
            print("Note: lm_head.weight parameter was skipped (due to tie_word_embeddings=True)")
        
        # 2. Add visual parameters from VLM (always keep)
        visual_added_count = 0
        visual_added_names = []  # Collect added visual parameter names
        
        # Directly iterate through all parameters in VLM state_dict, find visual-related parameters
        for name, param in vlm_state_dict.items():
            # Check if parameter name contains visual (whether model.visual or direct visual)
            if "visual" in name:
                # Convert parameter name: model.visual.xxx -> visual.xxx
                converted_name = self._convert_vlm_to_llm_param_name(name)
                merged_state_dict[converted_name] = param.detach().clone().contiguous()
                visual_added_count += 1
                visual_added_names.append(f"{name} -> {converted_name}")

        print(f"Added {visual_added_count} visual parameters from VLM:")
        for i, name in enumerate(visual_added_names, 1):
            print(f"  {i:3d}. {name}")
        
        # 3. Add other parameters to keep from VLM (VLM-specific non-visual non-language-model parameters)
        vlm_kept_count = 0
        vlm_kept_names = []  # Collect kept VLM parameter names
        for name, param in vlm_state_dict.items():
            # Skip visual parameters (already processed)
            if "visual" in name:
                continue
            # Skip language_model parameters (these will be obtained from LLM)
            if "language_model" in name:
                continue
                
            normalized_name = self._normalize_parameter_name(name)
            
            if normalized_name not in replace_set and normalized_name not in del_set:
                merged_state_dict[name] = param.detach().clone().contiguous()
                vlm_kept_count += 1
                vlm_kept_names.append(name)  # Record parameter name
        
        print(f"Kept {vlm_kept_count} other parameters from VLM:")
        for i, name in enumerate(vlm_kept_names, 1):
            print(f"  {i:3d}. {name}")
        
        # Only release LLM model, keep VLM instance for saving
        del llm, llm_state_dict, vlm_state_dict
        
        return merged_state_dict
    
    def _analyze_config_differences(self) -> Dict:
        """
        Analyze and merge configuration file differences between two models
        
        Returns:
            Merged configuration dictionary
        """
        print("\nStarting config file difference analysis...")
        
        # Get configuration key sets
        vlm_keys = set(self.vlm_config.keys())
        llm_keys = set(self.llm_config.keys())
        
        # Analyze key differences
        common_keys = vlm_keys & llm_keys  # Common keys
        vlm_only_keys = vlm_keys - llm_keys  # VLM-specific keys
        llm_only_keys = llm_keys - vlm_keys  # LLM-specific keys
        
        print(f"Number of common config keys: {len(common_keys)}")
        print(f"Number of VLM-specific config keys: {len(vlm_only_keys)}")
        print(f"Number of LLM-specific config keys: {len(llm_only_keys)}")
        
        # Define special handling keys (use VLM values)
        vlm_priority_keys = {"model_type", "architectures", "rope_scaling"}
        
        # Analyze value differences in common keys
        same_value_keys = []
        different_value_keys = []
        vlm_priority_different_keys = []  # Different keys that need VLM values
        
        for key in common_keys:
            vlm_value = self.vlm_config[key]
            llm_value = self.llm_config[key]
            
            if vlm_value == llm_value:
                same_value_keys.append(key)
            else:
                if key in vlm_priority_keys:
                    vlm_priority_different_keys.append(key)
                else:
                    different_value_keys.append(key)
        
        # Print detailed analysis results
        print(f"\n=== Config Key Analysis Results ===")
        print(f"Common config keys: {sorted(list(common_keys))}")
        
        print(f"\n--- Common keys with same values ({len(same_value_keys)} items) ---")
        for key in sorted(same_value_keys):
            print(f"  {key}: {self.vlm_config[key]}")
        
        print(f"\n--- Common keys with different values ({len(different_value_keys)} items) ---")
        for key in sorted(different_value_keys):
            print(f"  {key}:")
            print(f"    VLM: {self.vlm_config[key]}")
            print(f"    LLM: {self.llm_config[key]} (will use this value)")
        
        print(f"\n--- Common keys with different values but using VLM values ({len(vlm_priority_different_keys)} items) ---")
        for key in sorted(vlm_priority_different_keys):
            print(f"  {key}:")
            print(f"    VLM: {self.vlm_config[key]} (will use this value)")
            print(f"    LLM: {self.llm_config[key]}")
        
        print(f"\n--- VLM-specific keys ({len(vlm_only_keys)} items) ---")
        for key in sorted(vlm_only_keys):
            print(f"  {key}: {self.vlm_config[key]} (will keep)")
        
        print(f"\n--- LLM-specific keys ({len(llm_only_keys)} items) ---")
        for key in sorted(llm_only_keys):
            print(f"  {key}: {self.llm_config[key]} (will keep)")
        
        # Create merged configuration
        merged_config = {}
        
        # 1. Add VLM-specific keys
        for key in vlm_only_keys:
            merged_config[key] = self.vlm_config[key]
        
        # 2. Add LLM-specific keys
        for key in llm_only_keys:
            merged_config[key] = self.llm_config[key]
        
        # 3. Add common keys (same values, add directly)
        for key in same_value_keys:
            merged_config[key] = self.vlm_config[key]
        
        # 4. Add different value keys that use VLM values
        for key in vlm_priority_different_keys:
            merged_config[key] = self.vlm_config[key]  # Use VLM values
        
        # 5. Add other different value keys (use LLM values)
        for key in different_value_keys:
            merged_config[key] = self.llm_config[key]  # Use LLM values
            
        # Add out_hidden_size2 field
        merged_config['vision_config']["out_hidden_size2"] = self.llm_config["hidden_size"]
        
        # Force set specific fields
        merged_config["model_type"] = "qwen2_5_vl_merge"
        merged_config["architectures"] = ["Qwen2_5_VL_MergeForConditionalGeneration"]
        
        print(f"\nForced settings:")
        print(f"  model_type: {merged_config['model_type']}")
        print(f"  architectures: {merged_config['architectures']}")
        
        print(f"\nConfig merge completed, merged config has {len(merged_config)} keys")
        return merged_config
    
    def _save_merged_config(self, merged_config: Dict, output_path: str) -> None:
        """
        Save merged configuration file
        
        Args:
            merged_config: Merged configuration dictionary
            output_path: Output path
        """
        config_path = os.path.join(output_path, "config.json")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save configuration file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(merged_config, f, indent=2, ensure_ascii=False)
        
        print(f"Merged configuration file saved to: {config_path}")

    def save_merged_model(self, merged_state_dict: Dict[str, torch.Tensor], 
                         output_path: str, max_shard_size: str = "5GB") -> None:
        """
        Save merged model
        
        Args:
            merged_state_dict: Merged state_dict
            output_path: Output path
            max_shard_size: Maximum shard size
        """
        if self.vlm is None:
            raise ValueError("VLM model not loaded, cannot save")
            
        print(f"Saving merged model to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # First analyze config differences and generate merged config (but don't save)
        merged_config = self._analyze_config_differences()
        
        # Save model in transformers format (will automatically save original config)
        print("Saving in transformers format...")
        self.vlm.save_pretrained(
            output_path,
            state_dict=merged_state_dict,
            safe_serialization=True,
            max_shard_size=max_shard_size
        )
        print("Model saved successfully")
        
        # After model saving is complete, overwrite auto-generated config file with our merged config
        self._save_merged_config(merged_config, output_path)
        
        # Release VLM instance after saving
        del self.vlm
        self.vlm = None

    def _copy_materials(self, output_path: str) -> None:
        """
        Copy all files from materials directory to output directory
        
        Args:
            output_path: Output path
        """
        materials_path = self.materials_path
        
        if not os.path.exists(materials_path):
            print(f"Materials directory doesn't exist: {materials_path}")
            return
        
        if not os.path.isdir(materials_path):
            print(f"Materials path is not a directory: {materials_path}")
            return
        
        print(f"\n=== Materials File Copy Stage ===")
        print(f"Copying files from materials directory to: {output_path}")
        
        copied_count = 0
        for item in os.listdir(materials_path):
            src_path = os.path.join(materials_path, item)
            dst_path = os.path.join(output_path, item)
            
            try:
                if os.path.isfile(src_path):
                    # Copy file
                    shutil.copy2(src_path, dst_path)
                    print(f"  Copied file: {item}")
                    copied_count += 1
                elif os.path.isdir(src_path):
                    # Copy directory
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                    print(f"  Copied directory: {item}")
                    copied_count += 1
            except Exception as e:
                print(f"  Copy failed {item}: {e}")
        
        print(f"Materials copy completed, copied {copied_count} items")

    def merge_and_save(self, output_path: str, max_shard_size: str = "5GB") -> None:
        """
        Execute complete merge and save workflow
        
        Args:
            output_path: Output path
            max_shard_size: Maximum shard size
        """
        print("Starting model merge workflow...")
        
        
        # Merge models
        print("\n=== Model Parameter Merge Stage ===")
        merged_state_dict = self.merge_models()
        
        # Initialize visual projection layer
        self._initialize_visual_projection(merged_state_dict)
        
        # Save model (including config file)
        print("\n=== Model Save Stage ===")
        self.save_merged_model(merged_state_dict, output_path, max_shard_size)
        
        # Copy files from materials directory
        self._copy_materials(output_path)
        
        print("Model merge workflow completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model merger tool - Merge VLM models with LLM models for architectural compatibility")
    
    parser.add_argument(
        "--vlm-path", 
        type=str, 
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="VLM model path or Hugging Face model name (default: Qwen/Qwen2.5-VL-72B-Instruct)"
    )
    
    parser.add_argument(
        "--llm-path", 
        type=str, 
        default="Qwen/Qwen3-8B",
        help="LLM model path or Hugging Face model name (default: Qwen/Qwen3-8B)"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="Qwen-Merge-VL-8B-base",
        help="Output path (default: Qwen-Merge-VL-8B-base)"
    )
    
    parser.add_argument(
        "--materials-path", 
        type=str, 
        default="./materials",
        help="Materials folder path (default: ./materials)"
    )
    
    parser.add_argument(
        "--max-shard-size", 
        type=str, 
        default="5GB",
        help="Maximum shard size (default: 5GB)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    print(f"VLM model path: {args.vlm_path}")
    print(f"LLM model path: {args.llm_path}")
    print(f"Output path: {args.output_path}")
    print(f"Materials path: {args.materials_path}")
    print(f"Maximum shard size: {args.max_shard_size}")
    
    # Create merger
    merger = ModelMerger(args.vlm_path, args.llm_path, args.materials_path)
    
    try:
        # Execute merge
        merger.merge_and_save(args.output_path, args.max_shard_size)
    except Exception as e:
        print(f"Error occurred during merge process: {e}")
        raise


if __name__ == "__main__":
    main() 