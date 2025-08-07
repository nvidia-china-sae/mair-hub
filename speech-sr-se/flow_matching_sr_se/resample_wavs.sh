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

# Usage: ./resample_wavs.sh /path/to/src /path/to/target 24000

SRC_DIR="$1"
TARGET_DIR="$2"
TARGET_SR="$3"

if [[ -z "$SRC_DIR" || -z "$TARGET_DIR" || -z "$TARGET_SR" ]]; then
  echo "Usage: $0 <source_dir> <target_dir> <sample_rate>"
  exit 1
fi

if ! command -v sox &> /dev/null; then
  echo "Error: sox is not installed."
  exit 1
fi

# Get list of all wav files
mapfile -t FILES < <(find "$SRC_DIR" -type f -iname "*.wav")
TOTAL=${#FILES[@]}

if [[ $TOTAL -eq 0 ]]; then
  echo "No WAV files found in $SRC_DIR"
  exit 0
fi

# Progress bar function
draw_progress() {
  local current=$1
  local total=$2
  local width=50
  local percent=$(( 100 * current / total ))
  local filled=$(( width * current / total ))
  local empty=$(( width - filled ))

  printf "\r["
  printf "%0.s#" $(seq 1 $filled)
  printf "%0.s-" $(seq 1 $empty)
  printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

# Process each file
i=0
for SRC_FILE in "${FILES[@]}"; do
  ((i++))
  REL_PATH="${SRC_FILE#$SRC_DIR/}"
  DEST_FILE="$TARGET_DIR/$REL_PATH"
  mkdir -p "$(dirname "$DEST_FILE")"
  sox "$SRC_FILE" -r "$TARGET_SR" "$DEST_FILE" 2>/dev/null
  draw_progress "$i" "$TOTAL"
done

echo -e "\nDone! $TOTAL files resampled."