import re
from pathlib import Path
from shutil import rmtree
from functools import partial
from contextlib import nullcontext

from beartype import beartype

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, random_split, Subset


from cfm_superresolution import ConditionalFlowMatcherWrapper
from utils import STFTMag
from data import get_dataloader
from optimizer import get_optimizer

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

from tqdm import tqdm
import random
import torchaudio
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import plot_tensor, save_plot_
from einops import rearrange
from scipy.signal import sosfiltfilt, cheby1, resample, resample_poly
from scipy.io.wavfile import write
import librosa 
import math
import os
import matplotlib.pyplot as plt

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/flowhigh.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0
    return int(results[-1])

class FLowHighTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        cfm_wrapper: ConditionalFlowMatcherWrapper,
        *,
        batch_size, dataset: Dataset, validset: Dataset,
        num_train_steps = None, num_warmup_steps = None, num_epochs = None,
        lr = 1e-4, initial_lr = 1e-5, grad_accum_every = 1, wd = 0., max_grad_norm = 0.5,
        valid_prepare = False, valid_frac = 0.05,
        random_split_seed = 53,
        log_every = 10, save_results_every = 100, save_model_every = 500, results_folder = './results',
        force_clear_prev_results = None, split_batches = False, drop_last = False, 
        accelerate_kwargs: dict = dict(),
        original_sampling_rate = None, 
        tensorboard_logger = SummaryWriter,
        downsampling : str,
        sampling_rates : list,
        cfm_method = str,
        weighted_loss = False,
        model_name = str
    ):
        super().__init__()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(
            kwargs_handlers = [ddp_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )
        self.cfm_wrapper = cfm_wrapper
        self.register_buffer('steps', torch.Tensor([0]))
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.optim = get_optimizer(cfm_wrapper.parameters(), lr = lr, wd = wd)
        self.lr = lr
        self.initial_lr = initial_lr

        # max grad norm
        self.max_grad_norm = max_grad_norm

        # create dataset
        self.ds = dataset

        # split for validation
        if valid_prepare:
            self.train_ds = self.ds
            self.valid_ds = validset
        else:
            if valid_frac > 0:
                train_size = int((1 - valid_frac) * len(self.ds))
                valid_size = len(self.ds) - train_size
                self.train_ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
                self.print(f'training with dataset of {len(self.train_ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')

        assert len(self.train_ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        assert exists(num_train_steps) or exists(num_epochs), 'either num_train_steps or num_epochs must be specified'

        if exists(num_epochs):
            self.num_train_steps = len(dataset) // batch_size * num_epochs
        else:
            self.num_train_steps = num_train_steps
        print('num_train_stpes: ',num_train_steps)
        
        self.scheduler = CosineAnnealingLR(self.optim, T_max=self.num_train_steps)
        self.num_warmup_steps = num_warmup_steps if exists(num_warmup_steps) else 0
        
        # dataloader
        self.dataloader = get_dataloader(self.train_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)
        self.valid_dataloader = get_dataloader(self.valid_ds, batch_size = 1, shuffle = True, drop_last = drop_last)
        # fixed_valid_indices = [0, 11, 17, 31, 59, 61, 79, 83, 107, 119, 131, 151]  
        # fixed_valid_subset = Subset(self.valid_ds, fixed_valid_indices)
        # self.valid_dataloader = get_dataloader(fixed_valid_subset, batch_size = 1, shuffle = False, drop_last = drop_last)
        
        # prepare with accelerator 
        (self.cfm_wrapper, 
         self.optim, 
         self.scheduler, 
         self.dataloader
        ) = self.accelerator.prepare(
        self.cfm_wrapper, 
        self.optim, 
        self.scheduler,
        self.dataloader
        )

        # dataloader iterators
        self.dataloader_iter = cycle(self.dataloader)
        self.valid_dataloader_iter = cycle(self.valid_dataloader)

        # log & save
        self.log_every = log_every
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)
        print("results_folder",self.results_folder)

        # Ask if the existing checkpoint should be deleted
        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        # Create a directory for saving results
        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        # Hyperparameters for accelerator
        acc_hps = {
            "num_train_steps": self.num_train_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "learning_rate": self.lr,
            "initial_learning_rate": self.initial_lr,
            "wd": wd
        }
        self.accelerator.init_trackers("flowhigh", config=acc_hps)
        self.original_sampling_rate = original_sampling_rate
        self.tensorboard_logger = tensorboard_logger
        self.downsampling = downsampling
        self.sampling_rates = sampling_rates
        self.eval_stft = STFTMag(nfft=cfm_wrapper.flowhigh.audio_enc_dec.n_fft,
                                 hop=cfm_wrapper.flowhigh.audio_enc_dec.hop_length,
                                 window_len=cfm_wrapper.flowhigh.audio_enc_dec.win_length)
        self.cfm_method = cfm_method
        self.weighted_loss = weighted_loss
        self.model_name = model_name

    def save_validset_txt(self):
        valid_data_paths = [self.ds[idx][1] for idx in self.valid_ds.indices]
        txt_path = self.results_folder / 'validation_dataset.txt'
        with open(txt_path, 'w') as f:
            for valid_data_path in valid_data_paths:
                f.write(f"{valid_data_path}\n")

        self.print(f"Validation dataset paths have been saved to {txt_path}")
    
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.cfm_wrapper),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        cfm_wrapper = self.accelerator.unwrap_model(self.cfm_wrapper)
        pkg = cfm_wrapper.load(path)

        self.optim.load_state_dict(pkg['optim'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.cfm_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr

    def train_step(self):
        steps = int(self.steps.item())
        total_steps_per_epoch = len(self.train_ds) // self.batch_size
        current_epoch = steps // total_steps_per_epoch + 1
        self.cfm_wrapper.train()
        # adjust the lr according to the schedule
        # warm up 
        if steps < self.num_warmup_steps:
            # apply warmup
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            # after warmup period, start to apply lr annealing
            self.scheduler.step()
            
        # logs
        logs = {}

        # training step
        # Start the batch loop!
        for grad_accum_step in range(self.grad_accum_every):
            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.cfm_wrapper) if not is_last else nullcontext

            # Downsampled audio 
            HR_wave, wav_length, up_cond, random_sr = next(self.dataloader_iter)    
            win_length = self.cfm_wrapper.flowhigh.audio_enc_dec.win_length
            hop_length = self.cfm_wrapper.flowhigh.audio_enc_dec.hop_length         
            mel_lengths = torch.ceil((wav_length - win_length) / hop_length + 1) 
            up_cond = up_cond / up_cond.abs().max(dim=1, keepdim=True).values 
               
            with self.accelerator.autocast(), context():
                # cfm_wrapper forward 
                loss = self.cfm_wrapper(HR_wave,
                                        cond = up_cond, 
                                        cond_lengths = mel_lengths, 
                                        cfm_method = self.cfm_method,
                                        random_sr = random_sr,
                                        weighted_loss= self.weighted_loss) 
                # backward
                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.cfm_wrapper.parameters(), self.max_grad_norm)
        self.optim.step() # parameter update
        self.optim.zero_grad() 
        current_lr = self.scheduler.get_last_lr()[0]

        # tensorboard logging
        if self.is_main and not (steps % 10): 
            
            self.tensorboard_logger.add_scalar('training/cfm_loss', loss/self.grad_accum_every, global_step = steps)
            self.tensorboard_logger.add_scalar('training/lr',current_lr,global_step = steps)
        
        # log
        if not steps % self.log_every:

            self.print(f"Epoch {current_epoch}, Step {steps}: loss: {logs['loss']:0.3f}")

        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often 
        # for distributed learning
        self.accelerator.wait_for_everyone()

        # validation
        if self.is_main and not (steps % self.save_results_every): 
            if steps!=0:
                unwrapped_model = self.accelerator.unwrap_model(self.cfm_wrapper)
                
                with torch.inference_mode():
                    unwrapped_model.eval()
                    
                    # # # # # # # # # #
                    # validation code #
                    # # # # # # # # # #
                    
        # save model every so often
        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'FLowHigh.{steps}.pt')
            self.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, mid_ckpt=None, log_fn = noop):
        
        if mid_ckpt is not None:
            if os.path.exists(mid_ckpt):
                self.load(mid_ckpt)  
                print(f"Resuming training from checkpoint {mid_ckpt}")
            else:
                print(f"No checkpoint found at {mid_ckpt}, starting training from scratch")        
        else:
            print(f"starting training from scratch...")   
        
        for _ in tqdm(range(int(self.steps.item()), self.num_train_steps), desc=f"Training Progress {self.model_name}"):
        # while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
        self.accelerator.end_training()
