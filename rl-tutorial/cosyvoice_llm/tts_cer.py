# Reward calculation – **Whisper server client**
# Uses external REST server (`whisper_server.py`) running on GPU 3
# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache 2.0 License.
"""Shaped reward for LLaSA-TTS.

Changes
-------
* Whisper inference is **off-loaded** to a dedicated server (default
  `http://localhost:8000`).  No GPU usage inside the RL workers.
* Worker sends only **speech token IDs + text** → minimal network payload.
* Response: `{ nll, transcript }` – we reuse both for reward.
"""

from __future__ import annotations

import os, re, warnings, json, time, argparse
from typing import List

import numpy as np
import requests
import torch
from jiwer import cer, wer
from pypinyin import lazy_pinyin, Style
from tn.chinese.normalizer import Normalizer as ZhNormalizer

def _parse_ids(token_str: str) -> List[int]:
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", token_str)]

SERVER = os.getenv("WHISPER_SERVER", "http://localhost:8001")
SCORE_URL = f"{SERVER.rstrip('/')}/score"
HEALTH_URL = f"{SERVER.rstrip('/')}/healthz"

# quick health cache to avoid hitting server each call
_last_health = 0.0
zh_tn_model = ZhNormalizer(remove_erhua=False, remove_interjections=False, remove_puncts=True, overwrite_cache=False)

def _check_server():
    global _last_health
    if time.time() - _last_health < 30:
        return
    try:
        requests.get(HEALTH_URL, timeout=2)
        _last_health = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {SERVER}: {e}")


def _remote_whisper(tokens: List[int], text: str):
    _check_server()
    payload = {"tokens": tokens, "text": text}
    r = requests.post(SCORE_URL, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()  # {nll, transcript}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    beta_c: float = 3.0,
    tau_n: float = 3.0,
    lambda_c: float = 0.6,
    lambda_n: float = 0.4,
    debug_dump: bool = False,
) -> float:
    """Return reward in [0,1] using remote Whisper."""

    ids = _parse_ids(solution_str)

    try:
        resp = _remote_whisper(ids, ground_truth)
        nll = float(resp["nll"])
        transcript = resp.get("transcript", "")
        nll = None
    except Exception as e:
        warnings.warn(f"Whisper server error: {e}; CER-only fallback")
        nll = None
        transcript = ""

    # CER utility
    # hyp = transcript if transcript else ground_truth  # in worst case CER=0
    hyp = transcript
    ground_truth = zh_tn_model.normalize(ground_truth)
    hyp = zh_tn_model.normalize(hyp)
    # remove space
    # ground_truth = ground_truth.replace(" ", "")
    # hyp = hyp.replace(" ", "")
    # upper to lower
    ground_truth = ground_truth.lower()
    hyp = hyp.lower()
    print(f"ground_truth: {ground_truth}")
    print(f"hyp: {hyp}")
    # get pinyin of ground_truth and hyp

    ground_truth_pinyin = lazy_pinyin(
        ground_truth,
        style=Style.TONE3,
        tone_sandhi=True,
        neutral_tone_with_five=True,
    )
    hyp_pinyin = lazy_pinyin(
        hyp,
        style=Style.TONE3,
        tone_sandhi=True,
        neutral_tone_with_five=True,
    )
    # c = float(cer(ground_truth, hyp))

    # compute per
    hyp_pinyin_str = ' '.join(hyp_pinyin)
    ground_truth_pinyin_str = ' '.join(ground_truth_pinyin)
    c = float(wer(ground_truth_pinyin_str, hyp_pinyin_str))
    print(f"ground_truth_pinyin_str: {ground_truth_pinyin_str}")
    print(f"hyp_pinyin_str: {hyp_pinyin_str}")
    print(f"c: {c}")
    # c_lists = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # beta_c_lists = [1, 2, 3]
    # for c_list in c_lists:
    #     for beta_c_list in beta_c_lists:
    #         print(f"c: {c_list}, beta_c: {beta_c_list}, reward: {1.0 - np.tanh(beta_c_list * c_list)}")
    cer_u = 1.0 - np.tanh(beta_c * c)

    # NLL utility
    if nll is not None:
        nll_u = float(np.exp(-nll / tau_n))
    else:
        nll_u = 1e-9

    #denom = lambda_c / cer_u + lambda_n / nll_u
    #reward = (lambda_c + lambda_n) / denom if denom > 0 else 0.0
    reward = cer_u

    # print(f"\033[92mCER: {c:.3f}, NLL: {nll}, transcript: {transcript}, Reward: {reward:.4f}\033[0m")
    return max(0.0, min(1.0, reward))

# CLI quick test
if __name__ == "__main__":
    import sys
    
    def get_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Test TTS CER scoring with data from JSONL file",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            "--input", "-i",
            type=str,
            default="slam/mair-hub/rl-tutorial/cosyvoice_llm/data/emilia_zh-cosy-tiny-test.jsonl",
            help="Path to input JSONL file"
        )
        
        parser.add_argument(
            "--max-samples", "-n",
            type=int,
            default=None,
            help="Maximum number of samples to process (default: all)"
        )
        
        parser.add_argument(
            "--no-interactive",
            action="store_true",
            help="Run in non-interactive mode (process all samples without prompts)"
        )
        
        parser.add_argument(
            "--beta-c",
            type=float,
            default=3.0,
            help="Beta parameter for CER utility calculation"
        )
        
        parser.add_argument(
            "--tau-n",
            type=float,
            default=3.0,
            help="Tau parameter for NLL utility calculation"
        )
        
        parser.add_argument(
            "--lambda-c",
            type=float,
            default=0.6,
            help="Lambda parameter for CER weight"
        )
        
        parser.add_argument(
            "--lambda-n",
            type=float,
            default=0.4,
            help="Lambda parameter for NLL weight"
        )
        
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        return parser.parse_args()
    
    def load_jsonl(file_path: str):
        """Load data from jsonl file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def code_to_solution_str(code_list: List[int]) -> str:
        """Convert code list to solution string format."""
        return ''.join([f"<|s_{code}|>" for code in code_list])
    
    # Parse command line arguments
    args = get_args()
    
    try:
        # Load data from jsonl file
        print(f"Loading data from: {args.input}")
        data_list = load_jsonl(args.input)
        print(f"Loaded {len(data_list)} samples")
        
        # Limit samples if specified
        if args.max_samples is not None:
            data_list = data_list[:args.max_samples]
            print(f"Processing first {len(data_list)} samples (limited by --max-samples)")
        
        # Process each sample
        for i, sample in enumerate(data_list):
            print(f"\n--- Sample {i+1}/{len(data_list)} ---")
            print(f"Index: {sample.get('index', 'unknown')}")
            print(f"Text: {sample['text']}")
            
            # Extract required fields
            code_list = sample['code']
            ground_truth = sample['text']
            data_source = sample.get('index', f'sample_{i}')  # Use index as data_source
            
            # Convert code list to solution string
            solution_str = code_to_solution_str(code_list)
            print(f"Solution tokens: {len(code_list)} tokens")
            if args.debug:
                print(f"Solution string: {solution_str}")
            else:
                print(f"Solution string preview: {solution_str[:100]}..." if len(solution_str) > 100 else f"Solution string: {solution_str}")
            
            # Call compute_score function
            try:
                score = compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=None,
                    beta_c=args.beta_c,
                    tau_n=args.tau_n,
                    lambda_c=args.lambda_c,
                    lambda_n=args.lambda_n,
                    debug_dump=args.debug
                )
                print(f"Final Score: {score:.4f}")
            except Exception as e:
                print(f"Error computing score: {e}")
            
            # Ask user if they want to continue (for interactive mode)
            if not args.no_interactive and i < len(data_list) - 1:
                try:
                    response = input("\nPress Enter to continue or 'q' to quit: ").strip().lower()
                    if response == 'q':
                        break
                except KeyboardInterrupt:
                    print("\nStopped by user")
                    break
        
        print(f"\nProcessed {min(i+1, len(data_list))} samples")
        
    except FileNotFoundError:
        print(f"Error: File not found - {args.input}")
        print("Please check the file path or use --input to specify correct path")
        print("Run with --help for usage information")
    except Exception as e:
        print(f"Error: {e}")
