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
import torch
import torch.nn.functional as F
import whisper


def forward(self, x: torch.Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    x = (x + self.positional_embedding[: x.shape[1], :]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x


def replace_whisper_encoder_forward():
    """
    This function monkey patches the forward method of the whisper encoder.
    To be called before the model is loaded, it changes whisper to process audio with any length < 30s.
    """
    whisper.model.AudioEncoder.forward = forward
