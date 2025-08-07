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

import json
from types import SimpleNamespace
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchaudio 
from torchinfo import summary
from data import AudioDataset

from cfm_superresolution import (
    FLowHigh,
    MelVoco,
    ConditionalFlowMatcherWrapper
)

from trainer import FLowHighTrainer

def load_config(config_path):
    with open(config_path, 'r') as file:
        config_dict = json.load(file)
    return json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to the config JSON file"
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    
    # hparams = load_config('configs/config.json')
    hparams = load_config(args.config)

    torch.manual_seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)
    
    print('Num of current cuda devices:', n_gpus)
    print('Initializing logger...')
    logger = SummaryWriter(log_dir=hparams.train.save_dir)
    
    
    print('Initializing data loaders...')
    if hparams.data.data_path.endswith(".txt"):
        print("using filelist")
        dataset= AudioDataset(
            filelist_path=hparams.data.data_path, 
            noise_folder=hparams.data.noise_path, 
            rir_folder=hparams.data.rir_path,
            noise_prob=hparams.data.noise_prob,
            rir_prob=hparams.data.rir_prob,
            downsampling = hparams.data.downsampling_method,
            min_value = hparams.data.downsample_min,
            max_value = hparams.data.downsample_max
        )
    else:
        dataset= AudioDataset(
            folder=hparams.data.data_path, 
            noise_folder=hparams.data.noise_path, 
            rir_folder=hparams.data.rir_path,
            noise_prob=hparams.data.noise_prob,
            rir_prob=hparams.data.rir_prob,
            downsampling = hparams.data.downsampling_method,
            min_value = hparams.data.downsample_min,
            max_value = hparams.data.downsample_max
        )
    
    if hparams.data.valid_path.endswith(".txt"):
        validset = AudioDataset(
            filelist_path=hparams.data.valid_path, 
            downsampling = hparams.data.downsampling_method,
            min_value = hparams.data.downsample_min,
            max_value = hparams.data.downsample_max,
            mode='valid'
        )
    else:
        validset = AudioDataset(
            folder=hparams.data.valid_path, 
            downsampling = hparams.data.downsampling_method,
            min_value = hparams.data.downsample_min,
            max_value = hparams.data.downsample_max,
            mode='valid'
        )

    sampling_rates = list(range(hparams.data.downsample_min, hparams.data.downsample_max + 1000, 1000))
    
    print(f'Initializing Mel vocoder...')
    audio_enc_dec_type = MelVoco(n_mels= hparams.data.n_mel_channels, 
                                 sampling_rate= hparams.data.samplingrate, 
                                 f_max= hparams.data.mel_fmax, 
                                 n_fft= hparams.data.n_fft, 
                                 win_length= hparams.data.win_length, 
                                 hop_length= hparams.data.hop_length,
                                 vocoder= hparams.model.vocoder, 
                                 vocoder_config= hparams.model.vocoderconfigpath,
                                 vocoder_path = hparams.model.vocoderpath
    )
    
    # audio_enc_dec_type = LinearVoco()
    # audio_enc_dec_type = SpecVoco()
        
    print('Initializing FLowHigh...')
    model = FLowHigh(
                    architecture=hparams.model.architecture,
                    dim_in= hparams.data.n_mel_channels, # Same with Mel-bins 
                    audio_enc_dec= audio_enc_dec_type,
                    dim= hparams.model.dim,
                    depth= hparams.model.n_layers, 
                    dim_head= hparams.model.dim_head, 
                    heads= hparams.model.n_heads,
                    )
    
    print('Initializing CFM Wrapper...')
    cfm_wrapper = ConditionalFlowMatcherWrapper(flowhigh= model,
                                                cfm_method= hparams.model.cfm_path,
                                                sigma= hparams.model.sigma)
    
    summary(cfm_wrapper)

    print('Initializing FLowHigh Trainer...')
    trainer = FLowHighTrainer(cfm_wrapper= cfm_wrapper,
                              batch_size= hparams.train.batchsize,
                              dataset= dataset,
                              validset= validset,
                              num_train_steps= hparams.train.n_train_steps,
                              num_warmup_steps= hparams.train.n_warmup_steps,
                              num_epochs=None,
                              lr= hparams.train.lr,
                              initial_lr= hparams.train.initial_lr,
                              log_every = hparams.train.log_every, 
                              save_results_every = hparams.train.save_results_every, 
                              save_model_every = hparams.train.save_model_every,
                              results_folder= hparams.train.save_dir,
                              random_split_seed = hparams.train.random_split_seed,
                              original_sampling_rate = hparams.data.samplingrate, 
                              downsampling= hparams.data.downsampling_method,
                              valid_prepare = hparams.data.valid_prepare,
                              sampling_rates = sampling_rates,
                              cfm_method = hparams.model.cfm_path,
                              weighted_loss = hparams.train.weighted_loss,
                              model_name = hparams.model.modelname,
                              tensorboard_logger = logger,
                              )
    
    print('Start training...')
    trainer.train()