#!/usr/bin/env python3
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Simple classifier example based on Hugging Face Pytorch ResNet model."""

import argparse
import io
import logging
from typing import Any, List
import numpy as np
import torch  # pytype: disable=import-error


from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pytriton.proxy.types import Request

from omnisense.models import OmniSenseVoiceSmall
logger = logging.getLogger("examples.multi_instance_sensevoice_pytorch.server")


class _InferFuncWrapper:
    """Wraps a single OmniSenseVoiceSmall model instance for Triton."""

    def __init__(self, device_id: int):
        self._model = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=False, device_id=device_id)

    @batch
    def __call__(self, WAV: np.ndarray, WAV_LENS: np.ndarray, LANGUAGE: np.ndarray, TEXT_NORM: np.ndarray):
        """
        WAV: np.ndarray, WAV_LENS: np.ndarray
        LANGUAGE: np.ndarray, TEXTNORM: np.ndarray for backward compatibility, not used
        See: https://github.com/modelscope/FunASR/tree/main/runtime/triton_gpu
        """
        logger.debug("WAV: %s, WAV_LENS: %s, shapes: %s %s", type(WAV), type(WAV_LENS), WAV.shape, WAV_LENS.shape)
        wavs = [WAV[i, :WAV_LENS[i, 0]] for i in range(len(WAV))]

        results = self._model.transcribe_single_batch(
            wavs,
            language="auto",
            textnorm="woitn",
        )
        texts = [result.text for result in results]
        transcripts = np.char.encode(np.array(texts).reshape(-1, 1), "utf-8")
        return {"TRANSCRIPTS": transcripts}


def _infer_function_factory(device_ids: List[int]):
    """Creates a list of inference functions, one for each requested device ID."""
    infer_funcs = []
    for device_id in device_ids:
        infer_funcs.append(_InferFuncWrapper(device_id=device_id))
    return infer_funcs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Batch size of request.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--number-of-instances",
        type=int,
        default=2,
        help="Number of model instances to load.",
        required=False,
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    triton_config = TritonConfig(
        http_port=8001,
        grpc_port=8002,
        metrics_port=8003,
    )

    device_ids = [0] * args.number_of_instances

    with Triton(config=triton_config) as triton:
        logger.info("Loading SenseVoice model on device ids: %s", device_ids)
        triton.bind(
            model_name="sensevoice",
            infer_func=_infer_function_factory(device_ids),
            inputs=[
                Tensor(name="WAV", dtype=np.float32, shape=(-1,)),
                Tensor(name="WAV_LENS", dtype=np.int32, shape=(-1,)),
                Tensor(name="LANGUAGE", dtype=np.int32, shape=(-1,)),
                Tensor(name="TEXT_NORM", dtype=np.int32, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="TRANSCRIPTS", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
                batcher=DynamicBatcher(max_queue_delay_microseconds=10000),  # 10ms
            ),
            strict=True,
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()
