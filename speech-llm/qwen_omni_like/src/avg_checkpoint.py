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

import argparse
import torch
from typing import Dict, List
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="average checkpoints")
    parser.add_argument(
        "--exp-dir", required=True, type=str, help="exp dir"
    )
    parser.add_argument(
        "--output-name", default="pytorch_model_avg.bin", type=str, help="output name"
    )
    args = parser.parse_args()
    return args


def average_checkpoints(
    filenames: List[Path], device: torch.device = torch.device("cpu")
) -> dict:
    """Average a list of checkpoints.
    The function is mainly used for deepspeed converted checkpoint averaging, which only include model state_dict.

    Args:
      filenames:
        Filenames of the checkpoints to be averaged. We assume all
        checkpoints are saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    Returns:
      Return a dict (i.e., state_dict) which is the average of all
      model state dicts contained in the checkpoints.
    """
    n = len(filenames)

    avg = torch.load(filenames[0], map_location=device)

    # Make sure tensors do not require gradients so that in-place operations are allowed
    for k in avg:
        avg[k] = avg[k].to(device).detach()

    # Perform averaging in a torch.no_grad() context to avoid autograd tracking
    with torch.no_grad():
        # Identify shared parameters. Two parameters are said to be shared
        # if they have the same data_ptr
        uniqued: Dict[int, str] = dict()

        for k, v in avg.items():
            v_data_ptr = v.data_ptr()
            if v_data_ptr in uniqued:
                continue
            uniqued[v_data_ptr] = k

        uniqued_names = list(uniqued.values())

        for i in range(1, n):
            state_dict = torch.load(filenames[i], map_location=device)
            for k in uniqued_names:
                avg[k] += state_dict[k].to(device)

        for k in uniqued_names:
            if avg[k].is_floating_point():
                avg[k] /= n
            else:
                avg[k] //= n

    return avg

if __name__ == "__main__":
    args = get_args()
    filenames = sorted(Path(args.exp_dir).glob("epoch-*/pytorch_model.bin"))
    # get the last 5 checkpoints
    filenames = filenames[-5:]
    print(f"averaging {len(filenames)} checkpoints, {filenames}")
    avg = average_checkpoints(filenames)
    torch.save(avg, Path(args.exp_dir) / args.output_name)