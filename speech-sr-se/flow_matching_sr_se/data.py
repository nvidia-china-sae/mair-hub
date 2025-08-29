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

from pathlib import Path
from functools import wraps
from einops import rearrange
from beartype import beartype
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import scipy
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, sosfiltfilt, resample_poly
import numpy as np
import random
import json
from scipy.signal import fftconvolve

# utilities

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# dataset functions

class AudioDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder = None,
        filelist_path = None,
        noise_folder = None,
        noise_prob = 0.0,
        rir_folder = None,
        rir_prob = 0.0,
        audio_extension = ".wav",
        mode = None,
        downsampling = str(),
        min_value = 16000,
        max_value = 16000
    ):
        super().__init__()
        self.folder = folder
        self.filelist_path = filelist_path
        self.noise_dir = noise_folder
        self.noise_prob = noise_prob
        self.rir_prob = rir_prob
        
        if folder is not None and folder != "":
            path = Path(folder)
            assert path.exists(), 'folder does not exist'
            files = list(path.glob(f'**/*{audio_extension}'))
            assert len(files) > 0, 'no files found'
            self.files = files
        elif filelist_path is not None and filelist_path != "":
            self.files = self.load_filelist(filelist_path)
            assert len(self.files) > 0, 'no files found'
        
        if noise_folder is not None and noise_folder != "":
            noise_path = Path(noise_folder)
            assert noise_path.exists(), 'noise folder does not exist'
            noise_files = list(noise_path.glob(f'**/*{audio_extension}'))
            assert len(noise_files) > 0, 'no files found'
            self.noise_files = noise_files
            print(f"num noise files {len(noise_files)}")
        else:
            self.noise_files = None

        if rir_folder is not None and rir_folder != "":
            rir_path = Path(rir_folder)
            assert rir_path.exists(), 'rir folder does not exist'
            rir_files = list(rir_path.glob(f'**/*{audio_extension}'))
            assert len(rir_files) > 0, 'no files found'
            self.rir_files = rir_files
            print(f"num rir files {len(rir_files)}")
        else:
            self.rir_files = None
            
        self.audio_extension = audio_extension
        self.downsampling = downsampling
        self.min_value = min_value
        self.max_value = max_value
        self.mode = mode
        
    def load_filelist(self, path):
        files = []
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            file = (line["src"], line["tgt"])
            files.append(file)
        return files

    def __len__(self):
        return len(self.files)
    
    def rms(self, wav):
        return np.sqrt(np.mean(wav ** 2) + 1e-8)
    
    def rms_normalize(self, wav, target_rms=0.1):
        current_rms = self.rms(wav)
        return wav * (target_rms / current_rms)
    
    def add_noise(self, x, y, snr_db_range=(0, 20), normalize_rms=False):
        Lx = len(x)
        Ly = len(y)
        
        if Ly < Lx:
            repeat_times = (Lx // Ly) + 2
            y = np.tile(y, repeat_times)
            Ly = len(y)
        start = random.randint(0, Ly - Lx)
        y_crop = y[start:start + Lx]
        
        if normalize_rms:
            x = self.rms_normalize(x, target_rms=0.1)
            y_crop = self.rms_normalize(y_crop, target_rms=0.1)
        
        snr_db = random.randint(*snr_db_range)
        snr_linear = 10 ** (-snr_db / 20)
        scale = self.rms(x) / (self.rms(y_crop) + 1e-8) * snr_linear
        z = x + scale * y_crop

        if np.isnan(z).any():
            print("NaN encountered! Using unscaled noise.")
            z = x + y_crop
        
        peak = np.max(np.abs(z))
        if peak > 1.0:
            z = z / (peak + 1e-8)
        
        return z

    def normalize_audio(self, y):
        rms = np.sqrt(np.mean(y ** 2) + 1e-8)
        return y / (rms + 1e-8)
    
    def add_rir(self, clean_wav, rir_wav):
        rir_wav = rir_wav / (np.abs(rir_wav).max() + 1e-8)
        y = fftconvolve(clean_wav, rir_wav, mode="full")
        y = y[:len(clean_wav)]
        y = self.normalize_audio(y) * np.sqrt(np.mean(clean_wav ** 2))
        return y

    def __getitem__(self, idx):
        if self.folder is not None:
            file = self.files[idx]
            src_file, tgt_file = file, file
        else:
            src_file, tgt_file = self.files[idx]
        
        if self.noise_files is not None:
            noise_file = random.choice(self.noise_files)
        else:
            noise_file = None

        if self.rir_files is not None:
            rir_file = random.choice(self.rir_files)
        else:
            rir_file = None
            
        src_wave, src_sr = librosa.load(src_file, sr=None, mono=True)
        tgt_wave, tgt_sr = librosa.load(tgt_file, sr=None, mono=True)

        if rir_file is not None and np.random.rand() < self.rir_prob:
            rir_wave, _ = librosa.load(rir_file, sr=src_sr, mono=True)
            src_wave = self.add_rir(src_wave, rir_wave)
        
        if noise_file is not None and np.random.rand() < self.noise_prob:
            noise_wave, _ = librosa.load(noise_file, sr=src_sr, mono=True)
            src_wave = self.add_noise(src_wave, noise_wave, snr_db_range=(5, 30), normalize_rms=False)
            
        if self.downsampling == 'none':
            assert src_sr == tgt_sr
            if len(src_wave) < len(tgt_wave):
                src_wave = np.pad(src_wave, (0, len(tgt_wave) - len(src_wave)), 'constant', constant_values=0)
            elif len(src_wave) > len(tgt_wave):
                src_wave = src_wave[:len(tgt_wave)]
        
            length = tgt_wave.shape[-1]
            
            if self.mode == 'valid':
                return torch.from_numpy(src_wave).float(), src_wave.shape[-1]
            
            src_wave = torch.from_numpy(src_wave.copy()).float()
            tgt_wave = torch.from_numpy(tgt_wave.copy()).float()
            
            return tgt_wave, length, src_wave, None

        elif self.downsampling == 'torchaudio':
            wave, sr = torchaudio.load(file) # [1, Time]
            wave = rearrange(wave, '1 ... -> ...') # [Time]
            length = wave.shape[-1]
            return wave, length

        elif self.downsampling == 'librosa':
            src_wave /= np.max(np.abs(src_wave))
            nyq = src_sr // 2
            # min_value = 4000
            # max_value = 23000
            step = 1000
            sampling_rates = list(range(self.min_value, self.max_value + step, step))
            random_sr = random.choice(sampling_rates)

            if random_sr != tgt_sr:
                if self.mode == 'valid':
                    order = 8
                    ripple = 0.05
                else:
                    order = random.randint(1, 11)
                    ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])

                highcut = random_sr // 2
                hi = highcut / nyq
                sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
                d_HR_wave = sosfiltfilt(sos, src_wave)
                down_cond = librosa.resample(d_HR_wave, src_sr, random_sr, res_type='soxr_hq')
                up_cond = librosa.resample(down_cond, random_sr, tgt_sr, res_type='soxr_hq')
            else:
                up_cond = src_wave

            if len(up_cond) < len(tgt_wave):
                up_cond = np.pad(up_cond, (0, len(tgt_wave) - len(up_cond)), 'constant', constant_values=0)
            elif len(up_cond) > len(tgt_wave):
                up_cond = up_cond[:len(tgt_wave)]

            length = tgt_wave.shape[-1]

            if self.mode == 'valid':
                return torch.from_numpy(src_wave).float(), src_wave.shape[-1]
            
            return torch.from_numpy(tgt_wave).float(), length, torch.from_numpy(up_cond).float(), random_sr
        
        elif self.downsampling == 'scipy':
            
            src_wave /= np.max(np.abs(src_wave))
            nyq = src_sr // 2
            # min_value = 4000
            # max_value = 23000
            step = 1000
            sampling_rates = list(range(self.min_value, self.max_value + step, step))
            random_sr = random.choice(sampling_rates)

            if random_sr != tgt_sr:
                if self.mode == 'valid':
                    order = 8
                    ripple = 0.05
                
                else:
                    order = random.randint(1, 11)
                    ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])

                highcut = random_sr // 2
                hi = highcut / nyq

                sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
                d_HR_wave = sosfiltfilt(sos, src_wave)
                down_cond = resample_poly(d_HR_wave, random_sr, src_sr)
                up_cond = resample_poly(down_cond, tgt_sr, random_sr)
            else:
                up_cond = src_wave

            if len(up_cond) < len(tgt_wave):
                up_cond = np.pad(up_cond, (0, len(tgt_wave) - len(up_cond)), 'constant', constant_values=0)
            elif len(up_cond) > len(tgt_wave):
                up_cond = up_cond[:len(tgt_wave)]
        
            length = tgt_wave.shape[-1]
            
            if self.mode == 'valid':
                return torch.from_numpy(src_wave).float(), src_wave.shape[-1]
            
            up_cond = torch.from_numpy(up_cond.copy()).float()
            tgt_wave = torch.from_numpy(tgt_wave.copy()).float()
            
            return tgt_wave, length, up_cond, random_sr

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)
        if is_one_data:
            data = fn(data)
            return (data,)
        outputs = []
        
        for index, datum in enumerate(zip(*data)):
            
            if index == 1: # length 
                output = torch.tensor(datum, dtype=torch.long)
            elif index == 2: # up_cond wav
                output = fn(datum)
            elif index == 3:
                output = list(datum)
            else:
                output = fn(datum)  
            outputs.append(output)
        return tuple(outputs)
    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, num_workers = 8, persistent_workers=True, **kwargs)
