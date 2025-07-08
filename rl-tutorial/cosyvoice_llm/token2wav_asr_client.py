# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import requests
import soundfile as sf
import json
import numpy as np
import argparse
import time
import asyncio
import aiohttp

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="token2wav_asr",
        choices=[
            "token2wav_asr"
        ],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--concurrent-job",
        type=int,
        default=10,
        help="Number of concurrent requests to send in parallel",
    )

    return parser.parse_args()

def prepare_request(
    tokens,
    token_lens,
):
    data = {
        "inputs":[
            {
                "name": "TOKENS",
                "shape": tokens.shape,
                "datatype": "INT32",
                "data": tokens.tolist()
            },
            {
                "name": "TOKEN_LENS",
                "shape": token_lens.shape,
                "datatype": "INT32",
                "data": token_lens.tolist(),
            }
        ]
    }

    return data

def load_jsonl(file_path: str):
    """Load data from jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# -----------------------------
# Async client helpers
# -----------------------------

async def process_sample(idx, total, sample, session, url, semaphore):
    """Send a single request to the inference server and log the response."""
    async with semaphore:
        # Prepare request body
        code_list = sample["code"]
        tokens = np.array(code_list, dtype=np.int32).reshape(1, -1)
        token_lens = np.array([[len(tokens[0])]], dtype=np.int32)
        data = prepare_request(tokens, token_lens)

        # Send HTTP POST
        async with session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
            params={"request_id": "0"},
        ) as rsp:
            result = await rsp.json()

        transcripts = result["outputs"][0]["data"]

        # Output summary (prints may interleave across tasks)
        print(f"\n--- Sample {idx}/{total} ---")
        print(f"Text: {sample['text']}")
        print(tokens.shape, token_lens.shape)
        print(result)
        print(transcripts)


async def main_async():
    args = get_args()

    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{args.model_name}/infer"

    # Load dataset
    data_list = load_jsonl("/workspace/slam/mair-hub/rl-tutorial/cosyvoice_llm/data/emilia_zh-cosy-tiny-test.jsonl")

    # Concurrency primitives
    semaphore = asyncio.Semaphore(max(1, args.concurrent_job))
    connector = aiohttp.TCPConnector(ssl=False)

    start_time = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                process_sample(i + 1, len(data_list), sample, session, url, semaphore)
            )
            for i, sample in enumerate(data_list)
        ]
        await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(main_async())