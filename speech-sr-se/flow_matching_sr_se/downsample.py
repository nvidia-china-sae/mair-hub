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

# MIT License

# Copyright (c) 2024 Jun-Hak Yun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import argparse

def downsample_audio_file(input_path, output_path, target_sr):
    audio, original_sr = sf.read(input_path)
    if original_sr == target_sr:
        print(f"Skipping {input_path}: already at {target_sr} Hz")
        sf.write(output_path, audio, target_sr)
        return

    factor = gcd(original_sr, target_sr)
    up = target_sr // factor
    down = original_sr // factor

    audio_downsampled = resample_poly(audio, up, down, axis=0)
    sf.write(output_path, audio_downsampled, target_sr)
    print(f"Processed: {input_path} -> {output_path}")

def batch_downsample(input_dir, output_dir, target_sr):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.flac', '.aiff', '.aif', '.aifc')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            downsample_audio_file(input_path, output_path, target_sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample all audio files in a directory using scipy.resample_poly.")
    parser.add_argument("input_dir", help="Directory containing input audio files")
    parser.add_argument("output_dir", help="Directory to save downsampled audio files")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sampling rate (e.g., 16000)")

    args = parser.parse_args()
    batch_downsample(args.input_dir, args.output_dir, args.target_sr)
