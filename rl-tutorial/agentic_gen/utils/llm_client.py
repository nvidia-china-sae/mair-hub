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
LLM API Client
Unified interface for large language model invocation
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from openai import OpenAI

from core.exceptions import LLMApiError, ConfigurationError


@dataclass
class LLMResponse:
    """LLM response data model"""
    content: str
    model: str
    usage: Dict[str, int]
    response_time: float
    metadata: Dict[str, Any]


class LLMClient:
    """Unified LLM API client"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize LLM client
        
        Args:
            config: LLM configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.provider = config.get("provider", "openai")
        
        # Initialize client
        self._init_clients()
    
    def _init_clients(self):
        """Initialize clients for each LLM provider"""
        try:
            if self.provider == "openai":
                self._init_openai_client()
            else:
                raise ConfigurationError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize LLM client: {e}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if not self.config.get("api_key"):
            raise ConfigurationError("OpenAI API key is required")
        
        # Create OpenAI client instance
        client_kwargs = {"api_key": self.config["api_key"]}
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]
            
        self.openai_client = OpenAI(**client_kwargs)
        self.openai_config = self.config
    
    def generate_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                response = self._openai_completion(
                    prompt, system_prompt, model, temperature, max_tokens, **kwargs
                )
            else:
                raise LLMApiError(f"Unsupported provider: {self.provider}")
            
            response_time = time.time() - start_time
            
            # Wrap response
            llm_response = LLMResponse(
                content=response.get("content", ""),
                model=response.get("model", ""),
                usage=response.get("usage", {}),
                response_time=response_time,
                metadata=response.get("metadata", {})
            )
            
            self.logger.debug(f"LLM completion generated in {response_time:.2f}s")
            return llm_response
            
        except Exception as e:
            self.logger.error(f"LLM completion failed: {e}")
            raise LLMApiError(f"LLM API call failed: {e}")
    
    def _openai_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI API call"""
        model = model or self.openai_config.get("model", "gpt-4")
        temperature = temperature or self.openai_config.get("temperature", 0.7)
        max_tokens = max_tokens or self.openai_config.get("max_tokens", 2000)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else {},
            "metadata": {"finish_reason": response.choices[0].finish_reason}
        }

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str = None,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Batch generation
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt
            **kwargs: Other parameters
            
        Returns:
            List of responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                self.logger.error(f"Batch generation failed for prompt: {e}")
                # You can choose to skip failed requests or raise an exception
                continue
        
        return responses
    
    def parse_json_response(self, response: LLMResponse) -> Any:
        """
        Parse JSON formatted response
        
        Args:
            response: LLM response
            
        Returns:
            Parsed JSON data
        """
        try:
            # Try direct parsing
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try extracting JSON from code block
            content = response.content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
                    return json.loads(json_str)
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
                    return json.loads(json_str)
            
            raise LLMApiError(f"Failed to parse JSON response: {content}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple token estimation, may need more accurate method in real projects
        return len(text.split()) * 1.3  # About 1.3x the number of English words
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "provider": self.provider,
            "model": self.config.get("model", "unknown"),
            "total_requests": getattr(self, "_total_requests", 0),
            "total_tokens": getattr(self, "_total_tokens", 0)
        } 