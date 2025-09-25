'''
 * The Recognize Anything Model (RAM)
 * Written by Xinyu Huang
'''
import json
import warnings

import numpy as np
import torch
from torch import nn
try:
    from .utils import SwinTransformer, BinaryLinear_adapscaling

    # from .utils import *
except:
    from utils import SwinTransformer, BinaryLinear_adapscaling

    # from .utils import *


import os

def create_model(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config = "config_swinM_224.json"
    model = ram(os.path.join(current_dir, config))
    return model


class biRAM(nn.Module):
    def __init__(self,config,
                 image_size=224):
        r""" 
        Binarization Version 
        The Recognize Anything Model (RAM) inference module.
        RAM is a strong image tagging model, which can recognize any common category with high accuracy.
        Described in the paper " Recognize Anything: A Strong Image Tagging Model" 
        
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
        """
        super().__init__()
        vision_config_path = config
        vision_config = read_json(vision_config_path)
        # print(image_size)
        # print(vision_config['image_res'])
        assert image_size == vision_config['image_res']
        # assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']
        
        self.visual_encoder = SwinTransformer(
            img_size=vision_config['image_res'],
            patch_size=4,
            in_chans=1,
            embed_dim=vision_config['embed_dim'],
            depths=vision_config['depths'],
            num_heads=vision_config['num_heads'],
            window_size=vision_config['window_size'],
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False)
        # self.conv1x1 = 
        # self.target_shape = (197, 512)
        self.image_proj = BinaryLinear_adapscaling(vision_width, 512)
        self.image_proj2 = BinaryLinear_adapscaling(197, 145)
        # self.image_proj2 = BinaryLinear_adapscaling(vision_width, 512)
        # 145,197
        # self.label_embed = nn.Parameter(torch.load(f'{CONFIG_PATH}/data/textual_label_embedding.pth',map_location='cpu').float())


    def forward(self, image):
            
        
        # bs = image.shape[0]
        image_embeds = self.image_proj(self.visual_encoder(image))
        # print(image_embeds.shape)
        image_embeds = self.image_proj2(image_embeds.permute(0,2,1)).permute(0,2,1)
        return image_embeds
        
def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)

def load_checkpoint_swinlarge(model, url_or_filename, config, kwargs):
   
    vision_config_path = config
    window_size = read_json(vision_config_path)['window_size']
    print('--------------')
    print(url_or_filename)
    print('--------------')
    
    if os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg


# load RAM pretrained model parameters
def ram(config, pretrained=None, **kwargs):
    model = biRAM(config=config, **kwargs)
    if pretrained:
        model, msg = load_checkpoint_swinlarge(model, pretrained, config, kwargs)   
        # print('vit:', kwargs['vit'])
#         print('msg', msg)
    return model

