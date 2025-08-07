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

#!/bin/bash

set -e

ALL=false
if [[ "$1" == "--all" ]]; then
  ALL=true
fi

mkdir -p data/libritts_r
mkdir -p data/dns_challenge/noise
mkdir -p data/dns_challenge/rir

LIBRITTS_BASE="https://www.openslr.org/resources/141"
LIBRITTS_FILES=(
  "train_clean_100.tar.gz"
  "dev_clean.tar.gz"
  "test_clean.tar.gz"
)

if $ALL; then
  LIBRITTS_FILES+=(
    "train_clean_360.tar.gz"
    "train_other_500.tar.gz"
    "dev_other.tar.gz"
    "test_other.tar.gz"
  )
fi

echo "Downloading LibriTTS-R..."
for file in "${LIBRITTS_FILES[@]}"; do
  url="$LIBRITTS_BASE/$file"
  echo "Downloading $file..."
  wget -c "$url" -P data/libritts_r
  echo "Extracting $file..."
  tar -xzf "data/libritts_r/$file" -C data/libritts_r
done

NOISE_BASE="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/noise_fullband"
NOISE_FILES=(
  "datasets_fullband.noise_fullband.audioset_000.tar.bz2"
)

if $ALL; then
  NOISE_FILES+=(
    "datasets_fullband.noise_fullband.audioset_001.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_002.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_003.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_004.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_005.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_006.tar.bz2"
    "datasets_fullband.noise_fullband.freesound_000.tar.bz2"
    "datasets_fullband.noise_fullband.freesound_001.tar.bz2"
  )
fi

echo "Downloading DNS-Challenge Noise..."
for file in "${NOISE_FILES[@]}"; do
  url="$NOISE_BASE/$file"
  echo "Downloading $file..."
  wget -c "$url" -P data/dns_challenge/noise
  echo "Extracting $file..."
  tar -xjf "data/dns_challenge/noise/$file" -C data/dns_challenge/noise
done

RIR_BASE="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
RIR_FILES=("datasets_fullband.impulse_responses_000.tar.bz2")

echo "Downloading DNS-Challenge RIR..."
for file in "${RIR_FILES[@]}"; do
  url="$RIR_BASE/$file"
  echo "Downloading $file..."
  wget -c "$url" -P data/dns_challenge/rir
  echo "Extracting $file..."
  tar -xjf "data/dns_challenge/rir/$file" -C data/dns_challenge/rir
done

echo "All done."