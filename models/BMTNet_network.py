
import torch.nn as nn

try:
    from SimpleUNet import SimpleUNet
    from MainBMTNet import MainBMTNet
except:
    from .SimpleUNet import SimpleUNet
    from .MainBMTNet import MainBMTNet

def create_model():

    img_channel = 1
    width = 32

    enc_blks = [4, 4, 2]
    middle_blk_num = 2
    dec_blks = [4, 4, 4]
    num_heads=[1,2,4]
    
    net = SimpleUNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, num_heads=num_heads)
    
    in_channel = 1
    dim = 32
    out_channel = 3
    net2 = MainBMTNet(in_nc=in_channel,out_nc=out_channel, config=[4, 4, 4, 4, 4, 4, 4], dim=dim)


    net = BMTNet(net, net2)
    
    
    return net


class BMTNet(nn.Module):
    def __init__(self, model_1, model_2):
        super(BMTNet, self).__init__()
        
        self.model1 = model_1
        self.model2 = model_2
    def forward(self, x, qb_map):
        q_fixed = self.model1(x)
        r = self.model2(q_fixed, qb_map)
        return (r, q_fixed)


