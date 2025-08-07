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
print(os.getcwd())

import math
import logging
from random import random
from functools import partial
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
import torchode as to
from torchdiffeq import odeint
from beartype import beartype
from beartype.typing import Optional
from einops import rearrange, repeat, reduce, pack, unpack
import torchaudio.transforms as T
from torchaudio.functional import resample
from librosa.filters import mel as librosa_mel_fn
from utils import sequence_mask
import numpy
import matplotlib.pyplot as plt
from modules import LearnedSinusoidalPosEmb, ConvPositionEmbed, Transformer, ConvNeXtBlock
from postprocessing import PostProcessing
from init_vocoder import init_bigvgan

LOGGER = logging.getLogger(__file__)
logging.basicConfig(filename='model_debug.log', level=logging.INFO)

# helper functions
def exists(val):
    return val is not None

def identity(t):
    return t

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def coin_flip():
    return random() < 0.5

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# mel helpers
mel_basis = {}
hann_window = {}

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# tensor helpers
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def reduce_masks_with_and(*masks):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

def interpolate_1d(t, length, mode = 'bilinear'):
    " pytorch does not offer interpolation 1d, so hack by converting to 2d "

    dtype = t.dtype
    t = t.float()

    implicit_one_channel = t.ndim == 2
    if implicit_one_channel:
        t = rearrange(t, 'b n -> b 1 n')

    t = rearrange(t, 'b d n -> b d n 1')
    t = F.interpolate(t, (length, 1), mode = mode)
    t = rearrange(t, 'b d n 1 -> b d n')

    if implicit_one_channel:
        t = rearrange(t, 'b 1 n -> b n')

    t = t.to(dtype)
    
    return t

def curtail_or_pad(t, target_length):
    length = t.shape[-2]

    if length > target_length:
        t = t[..., :target_length, :]
    elif length < target_length:
        t = F.pad(t, (0, 0, 0, target_length - length), value = 0.)

    return t

# mask construction helpers
def mask_from_start_end_indices(seq_len: int, start: Tensor, end: Tensor):
    assert start.shape == end.shape
    device = start.device

    seq = torch.arange(seq_len, device = device, dtype = torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), seq_len)
    seq = seq.expand(*start.shape, seq_len)

    mask = seq >= start[..., None].long() # start 
    mask &= seq < end[..., None].long()
    
    return mask

def mask_from_frac_lengths(seq_len: int, frac_lengths: Tensor):
    device = frac_lengths.device

    lengths = (frac_lengths * seq_len).long() 
    max_start = seq_len - lengths 

    rand = torch.zeros_like(frac_lengths, device = device).float().uniform_(0, 1) 
    start = (max_start * rand).clamp(min = 0)
    end = start + lengths 
    
    return mask_from_start_end_indices(seq_len, start, end)

def mask_for_freqency(cond, batch: int, seq_len: int, mel_dim: int, device):

    for i in range(batch):
        
        import random
        mask_height = random.randint(10,20)
        rand_start = random.randint(20, mel_dim - mask_height) 
        minimum = torch.min(cond)
        cond[i, :, rand_start: rand_start + mask_height] = minimum + 1e-3

    return cond
    
    
# encoder decoders

class AudioEncoderDecoder(nn.Module):
    pass

class MelVoco(AudioEncoderDecoder): 
    def __init__(
        self,
        *,
        log = True,
        n_mels = 256,
        sampling_rate = 48000,
        f_max = 24000,
        f_min = 20,
        n_fft = 2048,
        win_length = 2048,
        hop_length = 480,
        vocoder = str,
        vocoder_config = './vocoder_config.json',
        vocoder_path = None
    ):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.f_min = f_min
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        
        if vocoder == 'bigvgan':
            self.vocoder_name = vocoder
            self.vocoder = init_bigvgan(vocoder_config, vocoder_path, vocoder_freeze=True)
        else:
            raise ValueError("unsuitable vocoder name")

    @property
    def downsample_factor(self):
        raise NotImplementedError

    @property
    def latent_dim(self):
        return self.n_mels
    
    def encode(self, audio):
        if torch.min(audio) < -1.:
            print('min value is ', torch.min(audio))
        if torch.max(audio) > 1.:
            print('max value is ', torch.max(audio))

        global mel_basis, hann_window
        if self.f_max not in mel_basis:
            mel = librosa_mel_fn(self.sampling_rate, self.n_fft, self.n_mels, self.f_min, self.f_max)
            mel_basis[str(self.f_max)+'_'+str(audio.device)] = torch.from_numpy(mel).float().to(audio.device)
            hann_window[str(audio.device)] = torch.hann_window(self.win_length).to(audio.device)

        audio = torch.nn.functional.pad(audio.unsqueeze(1), (int((self.n_fft-self.hop_length)/2), int((self.n_fft-self.hop_length)/2)), mode='reflect')
        audio = audio.squeeze(1)

        # complex tensor as default, then use view_as_real for future pytorch compatibility
        spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=hann_window[str(audio.device)],
                        center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(mel_basis[str(self.f_max)+'_'+str(audio.device)], spec)
        spec = spectral_normalize_torch(spec)
        spec = rearrange(spec, 'b d n -> b n d')
        return spec    

    def encode_torchaudio(self, audio):

        stft_transform = T.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            window_fn = torch.hann_window
        ).cuda()

        audio = audio.cuda()
        spectrogram = stft_transform(audio)

        mel_transform = T.MelScale(
            n_mels = self.n_mels,
            sample_rate = self.sampling_rate,
            n_stft = self.n_fft // 2 + 1,
            f_max = self.f_max
        ).cuda()

        spec = mel_transform(spectrogram)
        
        if self.log:
            spec = T.AmplitudeToDB()(spec)
        spec = rearrange(spec, 'b d n -> b n d')
        return spec

    def decode(self, mel):
        mel = rearrange(mel, 'b n d -> b d n')

        # if self.log:
        #     mel = DB_to_amplitude(mel, ref = 1., power = 0.5)

        if self.vocoder_name == 'bigvgan':  
            return self.vocoder.forward(mel)

    def decode_griffin_lim(self, mel_spec, n_iter=60):
        """
        Convert mel spectrogram (from encode()) back to waveform using Griffin-Lim.
        Args:
            mel_spec: (B, N, D) torch.Tensor — output of encode()
            n_iter: Number of Griffin-Lim iterations
        Returns:
            waveform: (B, T)
        """
        device = mel_spec.device
        mel_spec = rearrange(mel_spec, 'b n d -> b d n')  # (B, D, N)

        # Undo dynamic range compression (exp)
        mel_spec = dynamic_range_decompression_torch(mel_spec)

        # Get mel basis and inverse
        key = str(self.f_max) + '_' + str(device)
        mel_basis_matrix = mel_basis[key]  # (n_mels, n_fft//2+1)
        inv_mel_basis = torch.pinverse(mel_basis_matrix)  # (n_fft//2+1, n_mels)

        # Convert mel back to linear spectrogram
        linear_spec = torch.matmul(inv_mel_basis, mel_spec)  # (B, n_fft//2+1, N)

        # Use Griffin-Lim for phase reconstruction
        hann = hann_window[str(device)]

        # Apply Griffin-Lim manually since torch.stft returns complex
        def griffin_lim(mag_spec):
            batch_waveform = []
            for i in range(mag_spec.size(0)):
                magnitude = mag_spec[i]  # (F, N)
                phase = torch.rand_like(magnitude) * 2 * torch.pi
                complex_spec = magnitude * torch.exp(1j * phase)

                for _ in range(n_iter):
                    audio = torch.istft(complex_spec, n_fft=self.n_fft,
                                        hop_length=self.hop_length,
                                        win_length=self.win_length,
                                        window=hann,
                                        center=True,
                                        normalized=False,
                                        onesided=True)
                    complex_spec = torch.stft(audio, n_fft=self.n_fft,
                                            hop_length=self.hop_length,
                                            win_length=self.win_length,
                                            window=hann,
                                            center=True,
                                            normalized=False,
                                            onesided=True,
                                            return_complex=True)
                    phase = complex_spec / (complex_spec.abs() + 1e-8)
                    complex_spec = magnitude * phase
                batch_waveform.append(audio)
            return torch.stack(batch_waveform)

        waveform = griffin_lim(linear_spec)
        return waveform

class FLowHigh(Module):
    def __init__(
        self,
        *,
        audio_enc_dec: Optional[AudioEncoderDecoder] = None,
        dim_in = None, # 256
        dim_cond_emb = 0,
        dim = 1024,
        depth = 24,
        dim_head = 64,
        heads = 16,
        ff_mult = 4,
        ff_dropout = 0.,
        time_hidden_dim = None,
        conv_pos_embed_kernel_size = 31,
        conv_pos_embed_groups = None,
        attn_dropout = 0.,
        attn_flash = False,
        attn_qk_norm = True,
        use_gateloop_layers = False,
        architecture = None,
    ):
        super().__init__()
        dim_in = default(dim_in, dim)
        
        self.architecture = architecture

        if self.architecture=='transformer':
            time_hidden_dim = default(time_hidden_dim, dim)
        elif self.architecture=='convnext':
            time_hidden_dim = default(time_hidden_dim, dim)
        else:
            raise ValueError("Choose approriate architecture")
        
        self.audio_enc_dec = audio_enc_dec 

        self.proj_in = nn.Identity()    
        
        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU()
        )

        self.dim_cond_emb = dim_cond_emb
        self.to_embed = nn.Linear(dim_in * 2 + dim_cond_emb, dim)    
        self.null_cond = nn.Parameter(torch.zeros(dim_in), requires_grad = False)
        self.conv_embed = ConvPositionEmbed(
            dim = dim,
            kernel_size = conv_pos_embed_kernel_size,
            groups = conv_pos_embed_groups
        )

        if self.architecture =='transformer':
        
            self.transformer = Transformer(
                dim = dim,
                depth = depth,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                ff_dropout = ff_dropout,
                attn_dropout= attn_dropout,
                attn_flash = attn_flash,
                attn_qk_norm = attn_qk_norm,
                adaptive_rmsnorm = True,
                adaptive_rmsnorm_cond_dim_in = time_hidden_dim,
                use_gateloop_layers = use_gateloop_layers
            )   

        elif self.architecture=='convnext':
            intermediate_dim = dim * 3
            num_layers = 8
            layer_scale_init_value = 1
            self.convnext = nn.ModuleList(
                [
                    ConvNeXtBlock(
                        dim=dim,
                        intermediate_dim=intermediate_dim,
                        layer_scale_init_value=layer_scale_init_value,
                        hidden_dim=time_hidden_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
    
        dim_out = dim_in
        self.to_pred = nn.Linear(dim, dim_out, bias = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def hz_to_mel(self,f):
        if isinstance(f, (list, numpy.ndarray)): 
            f = numpy.array(f) 
        return 2595 * numpy.log10(1 + f / 700)

    def mel_bin_index(self, frequency, sample_rate, num_mel_bins):
        nyquist = sample_rate / 2
        m_min = self.hz_to_mel(0)
        m_max = self.hz_to_mel(nyquist)
        mel_value = self.hz_to_mel(frequency)
        bin_index = numpy.floor((mel_value - m_min) / (m_max - m_min) * num_mel_bins)
        if isinstance(bin_index, numpy.ndarray):
            bin_index = bin_index.astype(int)  
        else:
            bin_index = int(bin_index) 
        return bin_index

    @torch.inference_mode()
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1.:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale    
    
    def forward(
        self,
        x,
        *,
        times,
        self_attn_mask = None,
        cond_drop_prob = 0.1,
        target = None,
        cond = None,
        cond_mask = None,
        cond_freq_masking = False,
        random_sr = None,
        weighted_loss = False,
        cutoff_bins = None,
    ):

        x = self.proj_in(x) 
        cond = default(cond, target)
        
        if exists(cond):
            cond = self.proj_in(cond) 

        # shapes
        batch, seq_len, cond_dim = cond.shape
        assert cond_dim == x.shape[-1]


        # auto manage shape of times, for odeint times
        if times.ndim == 0:
            times = repeat(times, '-> b', b = cond.shape[0]) 
        if times.ndim == 1 and times.shape[0] == 1:
            times = repeat(times, '1 -> b', b = cond.shape[0]) 

        # Cond frequency masking 
        if cond_freq_masking:
            if self.training:
                cond = mask_for_freqency(cond, batch, seq_len, cond_dim, device=self.device)
            else:
                cond_freq_mask = torch.ones((batch, seq_len,cond_dim), device = cond.device, dtype =torch.bool)
                cond = cond * cond_freq_mask
        else:
            pass         

        # Classifier free guidance 
        if cond_drop_prob > 0.:
            cond_drop_mask = prob_mask_like(cond.shape[:1], cond_drop_prob, self.device)
            cond = torch.where(
                rearrange(cond_drop_mask, '... -> ... 1 1'),
                self.null_cond,
                cond
            )
                      
        # x.shape : [B, Time, channel]
        # cond.shape : [B, Time, channel]
        to_concat = [*filter(exists, (x,cond))] 
        
        # embed.shape : [B, Time, dim_in*2 ]
        embed = torch.cat(to_concat, dim = -1) 
        
        x = self.to_embed(embed)
        x = self.conv_embed(x, mask = self_attn_mask) + x

        time_emb = self.sinu_pos_emb(times)
        
        if self.architecture=='transformer':
            x = self.transformer(x, mask = self_attn_mask, adaptive_rmsnorm_cond = time_emb)        

        elif self.architecture=='convnext':
            x = x.transpose(1,2)
            for convnext_block in self.convnext:
                x = convnext_block(x, cond=time_emb)
        
            x = x.transpose(1,2)
            x = self.final_layer_norm(x)

        # Protect NaN
        logging.info(f"After transformer: {x}")
        if torch.isnan(x).any():
            print(x)
            logging.error("NaN detected after main architecture")
            
        x = self.to_pred(x)
                
        # Protect NaN
        logging.info(f"After predict: {x}")
        if torch.isnan(x).any():
            print(x)
            logging.error("NaN detected after last projection layer")


        # if no target passed in, just return logits
        # for inference mode
        if not exists(target):

            return x

        loss_mask = reduce_masks_with_and(cond_mask, self_attn_mask)

        if not exists(loss_mask):
            
            if weighted_loss == False:
                return F.mse_loss(x, target)
            
            elif weighted_loss == True:
                low_weight = 1.0
                high_weight = 2.0
                n_mels = self.audio_enc_dec.n_mels
                weight = torch.ones(batch,n_mels) * low_weight
                if isinstance(cutoff_bins, numpy.ndarray):
                    for i, bin_idx in enumerate(cutoff_bins):
                        weight[i, bin_idx:] = high_weight
                else:
                    exit()
                    weight[cutoff_bins:] = high_weight
                        
                weight = weight.unsqueeze(1).expand(batch, seq_len, n_mels).cuda()
                mse_loss = F.mse_loss(x, target, reduction='none') 
                weighted_mse_loss = mse_loss * weight
                mean_loss = weighted_mse_loss.mean()
                return mean_loss

        loss = F.mse_loss(x, target, reduction = 'none')
        loss = reduce(loss, 'b n d -> b n', 'mean')
        loss = loss.masked_fill(~loss_mask, 0.)

        # masked mean
        num = reduce(loss, 'b n -> b', 'sum')
        den = loss_mask.sum(dim = -1).clamp(min = 1e-5)
        loss = num / den
        return loss.mean()

def is_probably_audio_from_shape(t):
    return exists(t) and (t.ndim == 2 or (t.ndim == 3 and t.shape[1] == 1))

class ConditionalFlowMatcherWrapper(Module):
    @beartype
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        cfm_method = 'basic_cfm',
        torchdiffeq_ode_method = 'midpoint',   # [euler, midpoint]
        torchode_method_klass = to.Tsit5,      
        cond_drop_prob = 0.
    ):
        super().__init__()
        self.sigma = sigma
        self.flowhigh = flowhigh
        self.cond_drop_prob = cond_drop_prob
        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass
        self.cfm_method = cfm_method
        self.odeint_kwargs = dict(
            atol = ode_atol,
            rtol = ode_rtol,
            method = torchdiffeq_ode_method
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path, strict = True):
        # return pkg so the trainer can access it
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    # For mel repalcement
    def locate_cutoff_freq(self, mel, percentile=0.9995):
        def find_cutoff(x, percentile=0.99):
            percentile = x[-1] * percentile
            for i in range(1, x.shape[0]):
                if x[-i] < percentile:
                    return x.shape[0] - i
            return 0

        magnitude = torch.abs(mel)
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        return find_cutoff(energy, percentile)

    def mel_replace_ops(self, samples, input, cutoff_melbins):
        result = torch.zeros_like(samples)
        for i in range(samples.size(0)):

            result[i][..., cutoff_melbins[i]:] = samples[i][..., cutoff_melbins[i]:]
            result[i][..., :cutoff_melbins[i]] = input[i][..., :cutoff_melbins[i]]
        return result, cutoff_melbins
    
    def mel_cutoff_bins(self, input):
        cutoff_melbins = [] 
        for i in range(input.size(0)):
            cutoff_melbin = self.locate_cutoff_freq(torch.exp(input[i]))
            cutoff_melbins.append(cutoff_melbin) 
        return cutoff_melbins


    @torch.inference_mode()
    def sample(
        self,
        *,
        cond = None,
        cond_mask = None,
        time_steps = 4,
        cond_scale = 1.,
        decode_to_audio = True,
        std_1 = None,
        std_2 = None,
        mel_pp = False,
        cfm_method = None 
    ):
        # import pdb
        # pdb.set_trace()
        if cfm_method not in ['basic_cfm','independent_cfm_adaptive', 'independent_cfm_constant', 'independent_cfm_mix']:
            cfm_method = self.cfm_method
            # raise ValueError("Do not define cfm_method variable for sample()")
            
        if cfm_method in ['independent_cfm_adaptive', 'independent_cfm_constant','independent_cfm_mix']:
            if std_1 is None or std_2 is None:
                std_1 = 1.0
                std_2 = self.sigma
            
        cond_is_raw_audio = is_probably_audio_from_shape(cond)

        if cond_is_raw_audio:
            assert exists(self.flowhigh.audio_enc_dec)
            
            self.flowhigh.audio_enc_dec.eval()
            cond = self.flowhigh.audio_enc_dec.encode(cond)

        self_attn_mask = None
        shape = cond.shape # [B, Time, Channel]
        batch = shape[0]
        cutoff_bins = self.mel_cutoff_bins(cond)
        
        # neural ode
        self.flowhigh.eval()
    
        # ode function
        def ode_fn(t, x, *, packed_shape = None): 
            if exists(packed_shape):
                x = unpack_one(x, packed_shape, 'b *')
                
            out = self.flowhigh.forward_with_cond_scale(
                x,
                times = t,
                cond = cond,
                cond_scale = cond_scale,
                cond_mask = cond_mask,
                self_attn_mask = self_attn_mask
            )

            if exists(packed_shape):
                out = rearrange(out, 'b ... -> b (...)')
            return out # out.shape : [1, Time, mel_channel]

        if cfm_method == 'basic_cfm': 
            y0 = torch.randn_like(cond).cuda()

        elif cfm_method == 'independent_cfm_adaptive':
            # y0 from intended prior
            epsilon = torch.randn_like(cond).cuda()
            y0 = cond*std_1 + epsilon*std_2

        elif cfm_method == 'independent_cfm_constant':
            # y0 from intended prior
            epsilon = torch.randn_like(cond).cuda()
            y0 = cond*std_1 + epsilon*std_2
    
        elif cfm_method == 'independent_cfm_mix':
            # y0 from intended prior
            epsilon = torch.randn_like(cond).cuda()
            y0_low = cond*std_1 + epsilon*std_2
            y0_high = epsilon
            y0, _ = self.mel_replace_ops(y0_high, y0_low, cutoff_bins)
            
        t = torch.linspace(0, 1, time_steps + 1, device = self.device).cuda()
        if not self.use_torchode:

            LOGGER.debug('sampling with torchdiffeq')
            trajectory = odeint(ode_fn, y0, t, **self.odeint_kwargs) # bottle neck
            sampled = trajectory[-1]
            
            # # trajectory plot
            # n = len(trajectory) 
            # for i in range(n):
            #     plt.figure(figsize=(12, 4)) 
            #     plt.imshow(numpy.rot90(trajectory[i].squeeze().cpu().numpy(), 1), aspect='auto', origin='upper', interpolation='none')  
            #     plt.colorbar() 
            #     plt.title(f'trajectory[{i}]') 
            #     plt.xlabel('X-axis')  
            #     plt.ylabel('Y-axis') 
                
            #     plt.savefig(f'__trajectory[{i}].png', dpi=300, bbox_inches='tight') 
            #     plt.close()  

        else:
            LOGGER.debug('sampling with torchode')
            t = repeat(t, 'n -> b n', b = batch)
            y0, packed_shape = pack_one(y0, 'b *')
            fn = partial(ode_fn, packed_shape = packed_shape)
            term = to.ODETerm(fn)
            step_method = self.torchode_method_klass(term = term)
            step_size_controller = to.IntegralController(
                atol = self.odeint_kwargs['atol'],
                rtol = self.odeint_kwargs['rtol'],
                term = term
            )
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            jit_solver = torch.compile(solver)
            init_value = to.InitialValueProblem(y0 = y0, t_eval = t)
            sol = jit_solver.solve(init_value)
            sampled = sol.ys[:, -1]
            sampled = unpack_one(sampled, packed_shape, 'b *')
            
        if mel_pp:
            sampled, cutoff_bins = self.mel_replace_ops(sampled, cond, cutoff_bins)
            
        if not decode_to_audio or not exists(self.flowhigh.audio_enc_dec):
            return sampled 
        
        return self.flowhigh.audio_enc_dec.decode(sampled)
        # return self.flowhigh.audio_enc_dec.decode_griffin_lim(sampled, n_iter=100)

    def forward(
        self,
        x1,
        *,
        mask = None,
        cond = None,
        cond_mask = None,
        cond_lengths = None,
        input_sampling_rate = None, 
        cond_freq_masking = False, 
        random_sr = None,
        weighted_loss = None, 
        cfm_method = None # not necessary
    ):
        
        if cfm_method not in ['basic_cfm','independent_cfm_adaptive' ,'independent_cfm_constant','independent_cfm_mix']:
            cfm_method = self.cfm_method
        
        batch, seq_len, dtype, sigma_min = *x1.shape[:2], x1.dtype, self.sigma
        input_is_raw_audio, cond_is_raw_audio = map(is_probably_audio_from_shape, (x1, cond))
        
        if any([input_is_raw_audio, cond_is_raw_audio]):
            assert exists(self.flowhigh.audio_enc_dec), 'audio_enc_dec must be set on FLowHigh to train directly on raw audio'
            audio_enc_dec_sampling_rate = self.flowhigh.audio_enc_dec.sampling_rate
            input_sampling_rate = default(input_sampling_rate, audio_enc_dec_sampling_rate)            
            
            with torch.no_grad():
                self.flowhigh.audio_enc_dec.eval()
                # Making Ground truth mel-spectrogram
                if input_is_raw_audio:
                    x1 = resample(x1, input_sampling_rate, audio_enc_dec_sampling_rate)
                    x1 = self.flowhigh.audio_enc_dec.encode(x1) # x1.shape : [B, Time, channel]

                # Making mel-spectrogram which are empty in high-freqeuncy information
                if exists(cond) and cond_is_raw_audio:
                    cond = resample(cond, input_sampling_rate, audio_enc_dec_sampling_rate)
                cond = self.flowhigh.audio_enc_dec.encode(cond) # cond.shape : [B, Time, channel]
     
        if x1.size(1) != cond.size(1):
            max_timelength = max(x1.size(1), cond.size(1))
            x1 = F.pad(x1, (0, 0,max_timelength - x1.size(1), 0 ))  
            cond = F.pad(cond, (0, 0, max_timelength - cond.size(1), 0))

        # main conditional flow logic is below
        times = torch.rand((batch,), dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        if cfm_method == 'basic_cfm':
            """
            probability path: N(t x1, 1 - (1 - sigma) t)
            mu_t: t * x1 
            sigma_t: 1 - (1 - sigma_min)t
            sigma_min = 1e-4

            sample x_t: sigma_t * x0 + t * x1 = (1 - (1 - sigma_min) * t) * x0 + t * x1             
            target vector field: u_t = (x1 - (1 - sigma_min) x_t) / (1 - (1 - sigma_min) t) = x1 - (1 - sigma_min) * x0 
            
            if sigma_min = 0, then basic_cfm same with rectified-flow from standard normal distribution N(0,I)
            """   
            # x0 is gaussian noise
            x0 = torch.randn_like(x1)   # [B, Time, channel]
            cutoff_bins = None
            
            sigma_t = (1 - (1 - sigma_min) *t )
            
            # sample xt = noisy speech (\psi_t (x_0|x_1))                        
            w = sigma_t * x0 + t * x1  # [B, Time, channel]
            # w = (1 - (1 - sigma_min) * t) * x0 + t * x1  # [B, Time, channel]
            
            # target vector field u_t
            flow = x1 - (1 - sigma_min) * x0  # [B, Time, channel]
            # flow = (x1 - (1 - sigma_min) * w) / (1- (1 - sigma_min) *t)  # [B, Time, channel]
            
        elif cfm_method == 'independent_cfm_adaptive':
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, 1 - (1 - sigma_min) t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t:  1 - (1 - sigma_min) t

            sample x_t: mean + sigma * eps = t * x1 + (1 - t) *x0 + sigma_t * epsilon  
            target vector field: u_t = { (x1-x0) - (1-sigma_min)(xt-x0) } / { 1 - (1 - sigma_min) t } = (x1-x0) - (1-sigma_min) * epsilon 

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """   
            
            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)
            
            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()
            
            cutoff_bins = None
            
            mu_t = t * x1 + (1 - t) * x0 
            sigma_t = (1 - (1 - sigma_min)*t)
            
            # sample xt 
            w = mu_t + sigma_t * epsilon
            
            # target vector field u_t
            flow = (x1-x0) - (1-sigma_min) * epsilon # { (x1-x0) - (1-sigma_min)*(w-x0) } / {1 - (1 - sigma_min)*t} # [B, Time, channel]            
            

        elif cfm_method == 'independent_cfm_constant':
            """
            q(z) = q(x0)q(x1)
            probability path: N(t * x1 + (1 - t) *x0, sigma_t)
            mu_t: t * x1 + (1 - t) *x0
            sigma_t: sigma_min (small enough)

            sample x_t: mean + sigma*eps(eps~N(0,I)) = t * x1 + (1 - t) *x0 + sigma_t * epsilon
            target vector field: u_t = x1 - x0

            if sigma_min = 0, then independent_cfm same with rectified-flow from arbitrary distribution q(x0)
            """   
            
            # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)
            
            cutoff_bins = None
            
            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()
            
            mu_t = t * x1 + (1 - t) * x0 
            sigma_t = sigma_min
            
            # sample xt 
            w = mu_t + sigma_t * epsilon
            
            # target vector field u_t
            flow = x1 - x0  # [B, Time, channel]
            

        elif cfm_method == 'independent_cfm_mix':
            """
            q(z) = q(x0)q(x1)
            probability path_high: N(    t * x1          , 1 - (1 - sigma) t)
            probability path_low : N(t * x1 + (1 - t) *x0,       sigma_min    )
            
            x0: x^mel_low

            sample x_t: 
            target vector field: u_t = 
            """   
            
            # # eps ~ N(0,I)
            epsilon = torch.randn_like(cond)
            
            # get cutoff mel bins of LR mel
            cutoff_bins = self.mel_cutoff_bins(cond)
            
            # x0 represents low resolution audio(mel-spectogram)
            x0 = cond.detach().clone()
            
            # sample xt_high
            mu_t_high = t * x1 
            sigma_t_high = (1 - (1 - sigma_min) * t)                      
            xt_high = mu_t_high + sigma_t_high * epsilon   # [B, Time, channel]
            
            # sample xt_low
            mu_t_low = t * x1 + (1 - t) * x0 
            sigma_t_low = sigma_min
            xt_low = mu_t_low + sigma_t_low * epsilon

            w, _ = self.mel_replace_ops(xt_high, xt_low, cutoff_bins)
        
            # target vector field u_t
            flow = torch.zeros_like(x1)
            flow_high = x1 - (1 - sigma_min) * epsilon
            flow_low = x1 - x0  # [B, Time, channel]＼
            for i, cutoff_bin in enumerate(cutoff_bins):
                flow[i][..., cutoff_bin:] = flow_high[i][..., cutoff_bin:]
                flow[i][..., :cutoff_bin] = flow_low[i][..., :cutoff_bin]
            
        # x1.shape = cond.shape = x0.shape = w.shape = flow.shape = [Batch, Time, mel_bin]
        
        # Training mode!
        self.flowhigh.train()

        # Cut a small segment of mel-spectrogram
        cond_lengths = cond_lengths.to(torch.int32)
        max_cond_lengths = x1.size(1)
        x_mask = sequence_mask(cond_lengths, max_cond_lengths).unsqueeze(1) 
        out_size = 2* self.flowhigh.audio_enc_dec.sampling_rate // self.flowhigh.audio_enc_dec.hop_length

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (cond_lengths - out_size).clamp(0)
            offset_ranges = list( zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            
            import random

            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(cond_lengths)

            w_cut = torch.zeros(w.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=w.dtype, device=w.device)
            flow_cut = torch.zeros(flow.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=flow.dtype, device=flow.device)
            cond_cut = torch.zeros(cond.shape[0], out_size, self.flowhigh.audio_enc_dec.n_mels, dtype=cond.dtype, device=cond.device)
            
            x_cut_lengths = []

            for i, (w_, flow_, cond_, out_offset_) in enumerate(zip(w, flow, cond, out_offset)):
                
                # w_.shape = flow_.shape = cond_.shape = [Time, channel]
                x_cut_length = out_size + (cond_lengths[i] - out_size).clamp(None, 0)
                x_cut_lengths.append(x_cut_length)
                
                cut_lower, cut_upper = out_offset_, out_offset_ + x_cut_length
                w_cut[i, :x_cut_length,: ] = w_[cut_lower:cut_upper,: ]
                flow_cut[i, :x_cut_length,: ] = flow_[cut_lower:cut_upper,: ]
                cond_cut[i, :x_cut_length,: ] = cond_[cut_lower:cut_upper,: ]

            x_cut_lengths = torch.LongTensor(x_cut_lengths)
            x_cut_mask = sequence_mask(x_cut_lengths).unsqueeze(1).to(x_mask)

            w = w_cut 
            flow = flow_cut
            cond = cond_cut
            x_mask = x_cut_mask

        # forward 
        loss = self.flowhigh(
            x = w,
            cond = cond,
            cond_mask = cond_mask,
            times = times,
            target = flow,
            self_attn_mask = mask,
            cond_drop_prob = self.cond_drop_prob,
            cond_freq_masking = cond_freq_masking,
            random_sr = random_sr,
            weighted_loss = weighted_loss,
            cutoff_bins = cutoff_bins 
        )
        return loss

