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

from torchaudio.transforms import Spectrogram, InverseSpectrogram
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


class PostProcessing:
    def __init__(self, rank):
        self.stft = Spectrogram(2048, hop_length=480, win_length=2048, power=None, pad_mode='constant').cuda(rank)
        self.istft = InverseSpectrogram(2048, hop_length=480, win_length=2048, pad_mode='constant').cuda(rank)

    def get_cutoff_index(self, spec, threshold=0.99):
        energy = torch.cumsum(torch.sum(spec.squeeze().abs(), dim=-1), dim=0) 
        threshold = energy[-1] * threshold
        for i in range(1, energy.size(0)):
            if energy[-i] < threshold:
                return energy.size(0) - i
        return 0
    
    def post_processing(self, pred, src, length):
        # import pdb
        # pdb.set_trace()
        # pred, src : [1, Time]
        assert len(pred.shape) == 2 and len(src.shape) == 2 
        
        spec_pred = self.stft(pred) # [1, Channel, Time]
        spec_src  = self.stft(src) # [1, Channel, Time]

        # energy cutoff of spec_src
        cr = self.get_cutoff_index(spec_src)

        # Replacement
        spec_result = torch.empty_like(spec_pred)
        min_time_dim = min(spec_pred.size(-1), spec_src.size(-1))
        
        spec_result = spec_result[:, :, :min_time_dim]
        spec_pred = spec_pred[:, :, :min_time_dim]
        spec_src = spec_src[:, :, :min_time_dim]
        
        spec_result[:,cr:, ...] = spec_pred[:, cr:, ...]
        spec_result[:, :cr, ...] = spec_src[:, :cr, ...]
        
        audio = self.istft(spec_result, length=length) 
        audio = audio / torch.abs(audio).max() * 0.99
        return audio
    
    def post_processing_with_phase(self, pred, src, length):
        # pred, src : [1, Time]
        assert len(pred.shape) == 2 and len(src.shape) == 2 
        
        spec_pred = self.stft(pred) # [1, Channel, Time]
        spec_src  = self.stft(src) # [1, Channel, Time]

        batch = spec_pred.shape[0]
        cr = self.get_cutoff_index(spec_src)        

        # Replacement
        spec_result = torch.empty_like(spec_pred)
        min_time_dim = min(spec_pred.size(-1), spec_src.size(-1))
    
        spec_result = spec_result[:, :, :min_time_dim]
        spec_pred = spec_pred[:, :, :min_time_dim]
        spec_src = spec_src[:, :, :min_time_dim]
        
        pred_mag = torch.abs(spec_pred[:, cr:, ...])
        src_phase = torch.angle(spec_src[:, :cr, ...])
        
        # Replicate phase information to match the dimensions of spec_pred
        num_repeats = (spec_pred.size(1) - cr) // cr + 1
        replicate_phase = src_phase.repeat(batch, num_repeats, 1)
        replicate_phase = replicate_phase[:, - (spec_pred.size(1) - cr):, ...]
        print(pred_mag.size())
        print(replicate_phase.size())

        x = torch.cos(replicate_phase)
        y = torch.sin(replicate_phase)
        
        spec_result[:, cr:, ...] = pred_mag * (x + 1j * y)
        spec_result[:, :cr, ...] = spec_src[:, :cr, ...]
        
        audio = self.istft(spec_result, length=length) 
        audio = audio / torch.abs(audio).max() * 0.99
        return audio, src_phase ,replicate_phase
        
    
    # For mel repalcement
    def _locate_cutoff_freq(self, stft, percentile=0.985):
        def _find_cutoff(x, percentile=0.95):
            percentile = x[-1] * percentile
            for i in range(1, x.shape[0]):
                if x[-i] < percentile:
                    return x.shape[0] - i
            return 0

        magnitude = torch.abs(stft)
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        return _find_cutoff(energy, percentile)

    def mel_replace_ops(self, samples, input):
        for i in range(samples.size(0)):
            cutoff_melbin = self._locate_cutoff_freq(torch.exp(input[i]))
            samples[i][..., :cutoff_melbin] = input[i][..., :cutoff_melbin]
        return samples
