import json
import torch
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
vocoder_module_path = os.path.join(parent_dir, 'vocoder')


sys.path.append(vocoder_module_path)

from vocoder.BIGVGAN.bigvgan.models import BigVGAN
from vocoder.BIGVGAN.bigvgan.env import AttrDict   


def init_bigvgan(config, checkpoint, vocoder_freeze=False):    

    with open(config) as f:
        h = AttrDict(json.load(f))

    vocoder = BigVGAN(h)
    checkpoint_dict = torch.load(checkpoint, map_location="cuda")
    vocoder.load_state_dict(checkpoint_dict['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    if vocoder_freeze == True:    
        for param in vocoder.parameters():
            param.requires_grad = False

    return vocoder
