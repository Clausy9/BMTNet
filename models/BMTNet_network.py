
import torch.nn as nn
import torch
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
    def forward(self, x):
        q_fixed = self.model1(x)
        r = self.model2(q_fixed)
        return (r, q_fixed)



if __name__ == '__main__':
   
    img_channel = 1
    qb_channel = 4
    print(img_channel)
    width = 32

    enc_blks = [4, 4, 2]
    middle_blk_num = 2
    dec_blks = [4, 4, 4]
    num_heads=[1,2,4]
    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    net = create_model().cuda()
    
    x = torch.randn((1, 1, 256, 256)).cuda()
    y = torch.randn((1, 4, 256, 256)).cuda()
    y2 = torch.randn((1, 3, 256, 256)).cuda()
    # x = net(x, y)
    # x = net(x)
    from ptflops import get_model_complexity_info
    from thop import profile
    model = net # 假设我们使用的是ResNet-50模型
    input = torch.randn(1, 1, 256, 256)  # 1 是 batch size，3 是通道数，224x224 是图像大小
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=model, 
                                      kwargs ={'x':x},
                                      output_as_string=False,
                                      output_precision=4)
    # m = model(x)
    # print(m.shape)
    print("%s FLOPs:%sG  MACs:%sG  Params:%sM \n" %("model", str(int(flops) / (1000 ** 3)), str(int(macs) / (1000 ** 3)), str(int(params) / (1000 ** 2))))

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    print(flops, params)
    print(f"GFLOPs: {flops / 1e9} G")
    print(x.shape)

    