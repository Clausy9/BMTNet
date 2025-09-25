from copy import deepcopy
import argparse
import os
from tqdm import tqdm 
from dataloader import Demosaic_test
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from thop import profile

parser = argparse.ArgumentParser(description='EasySR')
parser.add_argument('--input_path', type=str, default='./input', help = 'input path')
parser.add_argument('--save_path', type=str, default='./output', help = 'save path')
parser.add_argument('--weights_path', type=str, default=None, help = 'model weights path')
from models.BMTNet_network import create_model

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def syn_qbmap(image, device):
    """
    synthsize Quad Bayer masks
    """
    _, _, h, w = image.shape
    mask1 = torch.zeros_like(image).to(device)
    mask2 = torch.zeros_like(image).to(device)
    mask3 = torch.zeros_like(image).to(device)
    
    mask1[..., 0:h:4, 0:w:4] = 1
    mask1[..., 0:h:4, 1:w:4] = 1
    mask1[..., 1:h:4, 0:w:4] = 1
    mask1[..., 1:h:4, 1:w:4] = 1
    
    mask2[..., 0:h:4, 2:w:4] = 2
    mask2[..., 0:h:4, 3:w:4] = 2
    mask2[..., 1:h:4, 2:w:4] = 2
    mask2[..., 1:h:4, 3:w:4] = 2
    
    mask2[..., 2:h:4, 0:w:4] = 2
    mask2[..., 2:h:4, 1:w:4] = 2
    mask2[..., 3:h:4, 0:w:4] = 2
    mask2[..., 3:h:4, 1:w:4] = 2
    
    mask3[..., 2:h:4, 2:w:4] = 3
    mask3[..., 2:h:4, 3:w:4] = 3
    mask3[..., 3:h:4, 2:w:4] = 3
    mask3[..., 3:h:4, 3:w:4] = 3
    

    return torch.cat([mask1,mask2,mask3], dim=1)


def coarse_inpaint(image, device):
    """
    coase inpaint implementation
    """
    
    _, _, h, w = image.shape
    mask = torch.zeros_like(image).to(device)
    mask[..., 1:h:4, 1:w:4] = 1
    mask[..., 3:h:4, 3:w:4] = 1
    

    # Create a mask to identify the valid pixels (non-zero in the mask)
    mask_indices = torch.nonzero(mask != 0)

    # Extract row and column indices of the valid pixels
    rows, cols = mask_indices[:, 2], mask_indices[:, 3]
    # Shift indices to get the indices of neighboring pixels
    top_indices = torch.stack([mask_indices[:, 0], mask_indices[:, 1], rows - 1, cols], dim=1)
    left_indices = torch.stack([mask_indices[:, 0], mask_indices[:, 1], rows, cols - 1], dim=1)
    topleft_indices = torch.stack([mask_indices[:, 0], mask_indices[:, 1], rows-1, cols - 1], dim=1)
    
    # Gather neighboring pixel values using the shifted indices
    top_values = image[top_indices[:, 0], top_indices[:, 1], top_indices[:, 2], top_indices[:, 3]]
    left_values = image[left_indices[:, 0], left_indices[:, 1], left_indices[:, 2], left_indices[:, 3]]
    topleft_values = image[topleft_indices[:, 0], topleft_indices[:, 1], topleft_indices[:, 2], topleft_indices[:, 3]]
    
    top_mask = top_values != 1023
    left_mask = left_values != 1023
    topleft_mask = topleft_values!=1023
    neighbors_mean = (torch.where(top_mask, top_values, torch.zeros_like(top_values)) +
                      torch.where(left_mask, left_values, torch.zeros_like(left_values))+\
                      torch.where(topleft_mask, topleft_values, torch.zeros_like(topleft_values))) / \
                     (top_mask.float() + left_mask.float()+ topleft_mask.float()+ 1e-8)

    filled_image = image.clone()
    filled_image[mask_indices[:, 0], mask_indices[:, 1], rows, cols] = neighbors_mean
    filled_image = filled_image.clip(0, 1023)
    return filled_image

if __name__ == '__main__':
    

    args = parser.parse_args()

    ## select active gpu devices
    device = None
    if torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_grad_enabled(False)
    
    ## definitions of model
    try:
        model = create_model() 
    except Exception:
        raise ValueError('not supported model type')
    
    model_path = args.weights_path
    if model_path is not None:
        # load test model
        print('load test model!')
        ckpt = torch.load(model_path, map_location=device)
        load_net = ckpt['model_state_dict']
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        model.load_state_dict(load_net)
    model = model.to(device)
    model = model.eval()
    
    root = args.save_path
    print(f"save images on {root}")
    os.makedirs(root, exist_ok=True)
    
    # create dataset for test
    test_set = Demosaic_test(args.input_path)
    test_dataloader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)


    for batch in tqdm(test_dataloader, ncols=80): 
        
        qbayer_sg = batch['qbayer_sg']
        _, _, h, w =qbayer_sg.shape
        qbayer_sg_name = batch['qbayer_sg_name']
        qbayer_sg = qbayer_sg.to(device)
        qb_map = syn_qbmap(qbayer_sg, qbayer_sg.device)
        qbayer_sg = coarse_inpaint(qbayer_sg, qbayer_sg.device)/1024
        output = model(qbayer_sg, qb_map)[0]
        output = output*256
        output = output.clamp(0, 255)
        qbayer_sg_name = qbayer_sg_name[0].split('/')[-1]
        output = output.detach().squeeze(0).squeeze(0).cpu().numpy()
        
        res_img = output.transpose(1, 2, 0)
        # save img
        res_img_path = root +'/'+ qbayer_sg_name[-8:].replace('.bin','.png')
        # print(res_img_path)
        imwrite(res_img, res_img_path)
        


        
       
       
        

        