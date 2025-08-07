import torch
# import torchaudio
import argparse
import numpy as np
# import csv
from cfm_superresolution import (
    MelVoco,
    FLowHigh,
    ConditionalFlowMatcherWrapper
)
import os 
from glob import glob 
from scipy.io.wavfile import write
# from utils import plot_tensor,save_plot,save_stft_plot
# import matplotlib.pyplot as plt
# from einops import rearrange
from postprocessing import PostProcessing
from tqdm import tqdm
import librosa
import scipy
from torchinfo import summary
# import time

def super_resolution(input_path, output_dir, target_sampling_rate, upsampling_method, cfm_wrapper, timestep, pp, cfm_method):
    
    extension = 'wav'
    audio_files = glob(os.path.join(input_path, f'*.{extension}'))
    
    print("Path of input data : ",input_path)
    print("Number of samples for speech super resolution : ",len(audio_files))
    print("method of cfm : ",cfm_method)
    
    with torch.no_grad():
        
        # super resolution
        for id, wav_file in enumerate(tqdm(audio_files, desc="Generating high-resolution audio files")):

            audio_file_name = os.path.basename(wav_file).replace('.wav','')
            save_dir = os.path.join(output_dir, f'{audio_file_name}.wav')
            
            # Up sampling the input audio
            if upsampling_method =='scipy':
                audio, sr = librosa.load(wav_file, sr=None, mono=True)
                cond = scipy.signal.resample_poly(audio, target_sampling_rate, sr)
                cond /= np.max(np.abs(cond))
                if isinstance(cond, np.ndarray):
                    cond = torch.tensor(cond).unsqueeze(0)
                cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T] 
                
            elif upsampling_method == 'librosa':
                audio, sr = librosa.load(wav_file, sr=None, mono=True)
                cond = librosa.resample(audio, sr, target_sampling_rate, res_type='soxr_hq')
                cond /= np.max(np.abs(cond))
                if isinstance(cond, np.ndarray):
                    cond = torch.tensor(cond).unsqueeze(0)
                cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T] 
            
            elif upsampling_method == 'none':
                audio, sr = librosa.load(wav_file, sr=None, mono=True)
                cond = audio
                cond /= np.max(np.abs(cond))
                if isinstance(cond, np.ndarray):
                    cond = torch.tensor(cond).unsqueeze(0)
                cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]
                print("no upsample")

            # reconstruct high resolution sample

            if cfm_method == 'basic_cfm':
                HR_audio = cfm_wrapper.sample(cond = cond, time_steps = timestep, cfm_method = cfm_method)
            elif cfm_method == 'independent_cfm_adaptive':
                HR_audio = cfm_wrapper.sample(cond = cond, time_steps = timestep, cfm_method = cfm_method, std_2 = 1.) 
            elif cfm_method == 'independent_cfm_constant':
                HR_audio = cfm_wrapper.sample(cond = cond, time_steps = timestep, cfm_method = cfm_method)
            elif cfm_method == 'independent_cfm_mix':
                HR_audio = cfm_wrapper.sample(cond = cond, time_steps = timestep, cfm_method = cfm_method) 

            HR_audio = HR_audio.squeeze(1) # [1, T]

            # post-proceesing w.r.s.t audio-level
            
            # do not use post processing because we want to do joint SE and SR
            # HR_audio_pp = pp.post_processing(HR_audio, cond, cond.size(-1)) # [1, T] 
            
            HR_audio_pp = HR_audio
            
            # save high resolution sample
            HR_audio_pp_npy = (HR_audio_pp.cpu().squeeze().clamp(-1,1).numpy()*32767.0).astype(np.int16) 
            write(save_dir, target_sampling_rate, HR_audio_pp_npy)          


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech super-resolution with CFM")
    parser.add_argument('--input_path', type=str, required=True, help="path of input low-resoltuion audio")
    parser.add_argument('--output_path', type=str, required=True, help="path of output high-resolution audio")
    parser.add_argument('--target_sampling_rate', type=int, required=True, help = 'sampling rate of original dataset')    
    parser.add_argument('--up_sampling_method', type=str, required=True, choices=['torchaudio','librosa','scipy', 'none'], help = 'upsamplingmethod')
    parser.add_argument('--architecture', type=str, required=True, choices=['transformer','convnext'], help = 'architecture')
    parser.add_argument('--time_step', type=int, required=False, default=4, help='number of timesteps for Solving ODE')
    parser.add_argument('--cfm_method', type=str, required=True, choices=['basic_cfm','independent_cfm_adaptive','independent_cfm_constant','independent_cfm_mix'], help = 'method of cfm')
    parser.add_argument('--ode_method', type=str, required=True, choices=['euler','midpoint'], help = 'method of solving ODE')
    parser.add_argument('--sigma', type=float, required=False, default=1e-4, help='standard deviation')     
    parser.add_argument('--model_path', type=str, required=True, help="path of pre-trained checkpoint of model")
    parser.add_argument('--n_layers', type=int, required=False, default=2, help='number of main architecture layers')
    parser.add_argument('--n_heads', type=int, required=False, default=16, help='number of heads for attention')
    parser.add_argument('--dim_head', type=int, required=False, default=64, help='number of dimension for MHA')
    parser.add_argument('--n_mels', type=int, required=False, default=256, help='number of mel bins')
    parser.add_argument('--f_max', type=int, required=False, default=24000, help='f_max')
    parser.add_argument('--n_fft', type=int, required=False, default=2048, help='n_fft')
    parser.add_argument('--win_length', type=int, required=False, default=2048, help='window length')
    parser.add_argument('--hop_length', type=int, required=False, default=480, help='hop length')
    parser.add_argument('--vocoder', type=str, required=True, choices=['bigvgan'], help="type of vocoder model")
    parser.add_argument('--vocoder_path', type=str, required=True, help = 'path of vocoder checkpoint')
    parser.add_argument('--vocoder_config_path', type=str, required=True, help = 'path of vocoder config')

    args = parser.parse_args()

    output_dir = args.output_path + '/output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # for post-processing
    pp = PostProcessing(0)

    print(f'Initializing FLowHigh...')
    audio_enc_dec_type_for_infer = MelVoco(n_mels= args.n_mels, 
                                           sampling_rate= args.target_sampling_rate, 
                                           f_max= args.f_max, 
                                           n_fft= args.n_fft, 
                                           win_length= args.win_length, 
                                           hop_length= args.hop_length,
                                           vocoder=args.vocoder, 
                                           vocoder_config= args.vocoder_config_path, 
                                           vocoder_path = args.vocoder_path
    )        
        
    # checkpoint load
    model_checkpoint = torch.load(args.model_path, map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    SR_generator = FLowHigh(
            dim_in = audio_enc_dec_type_for_infer.n_mels, 
            audio_enc_dec = audio_enc_dec_type_for_infer,
            depth = args.n_layers,
            dim_head = args.dim_head,
            heads = args.n_heads,
            architecture = args.architecture,)

    cfm_wrapper=ConditionalFlowMatcherWrapper(flowhigh=SR_generator, 
                                                cfm_method = args.cfm_method, 
                                                torchdiffeq_ode_method=args.ode_method, 
                                                sigma = args.sigma)
    cfm_wrapper.load_state_dict(model_checkpoint['model']) # dict_keys(['model', 'optim', 'scheduler'])

    print(f'Setting the model to evalutaion mode ...')       
    SR_generator = SR_generator.cuda().eval()
    cfm_wrapper = cfm_wrapper.cuda().eval()

    number = sum(p.numel() for p in cfm_wrapper.parameters() if p.requires_grad)
    if number >= 1_000_000:
        print(f"Total number of parameters: {number / 1_000_000:.2f} million") 
    elif number >= 1_000:
        print(f"Total number of parameters: {number / 1_000:.2f} thousand") 
    else:
        print(f"Total number of parameters: {str(number)}")

    summary(cfm_wrapper)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print(f'Start super resolution ...')   
    super_resolution(
                        args.input_path, 
                        output_dir, 
                        args.target_sampling_rate,
                        args.up_sampling_method,
                        cfm_wrapper,
                        args.time_step,
                        pp,
                        args.cfm_method,
                        )
