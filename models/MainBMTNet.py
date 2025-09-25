# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import rff
import numbers
from typing import Optional, Callable
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat

def create_model(args):

    if args.in_channel:
        in_channel = args.in_channel
    else:
        in_channel = 1
        
    if args.dim:
        dim = 32
    else:
        dim = 64
    
    
    out_channel = 3
    net = MainBMTNet(in_nc=in_channel,out_nc=out_channel, config=[4, 4, 4, 4, 4, 4, 4], dim=dim)
    
    return net


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

'''
Modify from
Bivit: Extremely compressed binary vision transformers
@inproceedings{he2023bivit,
  title={Bivit: Extremely compressed binary vision transformers},
  author={He, Yefei and Lou, Zhenyu and Zhang, Luoming and Liu, Jing and Wu, Weijia and Zhou, Hong and Zhuang, Bohan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5651--5663},
  year={2023}
}
'''
class BiLinear(nn.Linear):
    def __init__(self,  *kargs, bias=True, quantize_act=True, weight_bits=1, input_bits=1, clip_val=2.5):
        super(BiLinear, self).__init__(*kargs,bias=True)
        self.quantize_act = quantize_act
        self.weight_bits = weight_bits

        self.init = True
        nn.init.uniform_(self.weight, -0.015, 0.015)
        if self.quantize_act:
            self.input_bits = input_bits
            self.act_quant_layer = BinaryQuantizer().apply
            self.weight_quant_layer = BinaryQuantizer().apply

        self.scaling_factor = nn.Parameter(
            torch.zeros((self.out_features, 1)), requires_grad=True
        )
        self.first_run = True

    def forward(self, input):
        if self.first_run:
            self.first_run = False
            if torch.sum(self.scaling_factor.data) == 0:
                # print('attn initing scaling_factor...')
                self.scaling_factor.data = torch.mean(abs(self.weight), dim=1, keepdim=True)

        if self.weight_bits == 1:
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            weight = self.scaling_factor * self.weight_quant_layer(real_weights)
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        if self.input_bits == 1:
            ba = self.act_quant_layer(input)
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        out = F.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x

'''
Modify from
Basic binary convolution unit for binarized image restoration network

@article{xia2022basic,
  title={Basic binary convolution unit for binarized image restoration network},
  author={Xia, Bin and Zhang, Yulun and Wang, Yitong and Tian, Yapeng and Yang, Wenming and Timofte, Radu and Van Gool, Luc},
  journal={arXiv preprint arXiv:2210.00405},
  year={2022}
}
'''
class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        # print(self.weight.dtype)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0) # STE
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding)

        return y

class BinaryConv2dSkip1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=False):
        super(BinaryConv2dSkip1x1, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups)

    def forward(self, x):
        out = self.move0(x)
        out = BinaryQuantize_Quad().apply(out)
        out = self.binary_conv(out)
        out =self.relu(out)
        out = out + self.conv_skip(x)
        return out

class BConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(BConv2d, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)

    def forward(self, x):
        out = self.move0(x)
        out = BinaryQuantize_Quad().apply(out)
        out = self.binary_conv(out)
        out = self.relu(out)
        out = out + x
        return out

class BinaryQuantize_Quad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
    def unsupported_flops(self, module, inputs, output):
        input = inputs[0]
        inputs_dims = list(input.shape)
        overall_flops = int(np.prod(inputs_dims))

        module.__flops__ += overall_flops


class SoftmaxBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        attn = input.softmax(dim=-1)
        thresh, idx = torch.max(attn, dim=-1) 
        thresh *= 0.25 
        thresh = thresh.unsqueeze(-1)
        out = torch.where(attn < thresh, 0, 1)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


'''
Modify from
VMamba: Visual State Space Model
@misc{liu2024vmambavisualstatespace,
      title={VMamba: Visual State Space Model}, 
      author={Yue Liu and Yunjie Tian and Yuzhong Zhao and Hongtian Yu and Lingxi Xie and Yaowei Wang and Qixiang Ye and Yunfan Liu},
      year={2024},
      eprint={2401.10166},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.10166}, 
}
'''

class BiSS2D(nn.Module):
    '''
    Binary version of Mamba, binarize all projections to reduce complexity.
    '''
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model 
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = BiLinear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = BConv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias
        )
        self.act = nn.SiLU()

        
        self.l1 = BiLinear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        self.l2 = BiLinear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        self.l3 = BiLinear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        self.l4 = BiLinear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)


        self.ld1, bias1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        self.ld2, bias2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        self.ld3, bias3 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        self.ld4, bias4 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs)
        # self.dt_projs = (self.ld1,self.ld2,self.ld3,self.ld4)
        # self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([bias1, bias2, bias3, bias4], dim=0))  # (K=4, inner)
        # print(self.ld4.bias.shape)
        # del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = BiLinear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = BiLinear(dt_rank, d_inner, bias=False)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        return dt_proj, inv_dt

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        xs2 = xs.view(B, K, -1, L)
        x0 = xs2[:, 0, :, :].view(B, L, -1)
        x1 = xs2[:, 1, :, :].view(B, L, -1)
        x2 = xs2[:, 2, :, :].view(B, L, -1)
        x3 = xs2[:, 3, :, :].view(B, L, -1)
        x0 = self.l1(x0)
        x1 = self.l2(x1)
        x2 = self.l3(x2)
        x3 = self.l4(x3)
        x_dbl = rearrange(torch.stack([x0,x1,x2,x3], dim=1),'b k l c -> b k c l')

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts2 = dts.view(B, K, -1, L)
        dts2 = [dts[:, i, :, :].contiguous().view(B, L, -1) for i in range(4)]
        dt0 = self.ld1(dts2[0])
        dt1 = self.ld2(dts2[1])
        dt2 = self.ld3(dts2[2])
        dt3 = self.ld4(dts2[3])

        dts = rearrange(torch.stack([dt0,dt1,dt2,dt3], dim=1),'b k l d -> b k d l')
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        # print(xs.shape)
        '''
        Core Selective Scan kept as Full Precision Format
        '''
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # print(out_y.shape)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class BiMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            # is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = BiSS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input):
        input = rearrange(input, 'b c h w -> b h w c')
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class BiWMSA(nn.Module):
    """ Self-attention module in Binary Cross Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(BiWMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = BiLinear(self.input_dim, 2*self.input_dim, bias=True)
        self.embedding_layer_qb = BiLinear(self.input_dim, self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = BiLinear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

        self.quant_layer = BinaryQuantizer().apply
        self.attn_quant_layer = SoftmaxBinaryQuantizer().apply


    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x, y):
        """ Forward pass of Binary Window Multi-head Self-attention module.
        Args:
            x: input image tensor with shape of [b h w c];
            y: input position tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': 
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
            y = torch.roll(y, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        kv = self.embedding_layer(x)
        k, v = rearrange(kv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(2, dim=0)
        y = rearrange(y, 'b (w1 p1) (w2 p2) c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        q = self.embedding_layer_qb(y)
        q = rearrange(q, 'b nw np (h c) -> h b nw np c', c=self.head_dim)
        
        binary_q = self.quant_layer(q)
        binary_k = self.quant_layer(k)
        binary_v = self.quant_layer(v)


        sim = torch.einsum('hbwpc,hbwqc->hbwpq', binary_q, binary_k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q') 
        
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
    
        probs = self.attn_quant_layer(sim).float().detach() - sim.softmax(-1).detach() + sim.softmax(-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, binary_v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class BiCSwin(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ 
        Binary Cross Swin Transformer Block
        """
        super(BiCSwin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = BiWMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            BiLinear(input_dim, 4 * input_dim),
            nn.GELU(),
            BiLinear(4 * input_dim, output_dim),
        )
        self.ln3 = nn.LayerNorm(input_dim)


    def forward(self, x, y):
        x = x + self.drop_path(self.msa(self.ln1(x), self.ln3(y)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class BMTBlock_enc(nn.Module):
    def __init__(self, mamba_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ Binary Mamba-Transformer Block
        """
        super(BMTBlock_enc, self).__init__()
        self.mamba_dim = mamba_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = BiCSwin(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = BConv2d(self.mamba_dim+self.trans_dim, self.mamba_dim+self.trans_dim, 1, bias=True)
        self.conv1_2 = BConv2d(self.mamba_dim+self.trans_dim, self.mamba_dim+self.trans_dim, 1, bias=True)

        self.mamba_block = BiMambaBlock(
                hidden_dim=self.mamba_dim,
                drop_path=0,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=16,
                expand=2)

    def forward(self, x, y):

        mamba_x, trans_x = torch.split(self.conv1_1(x), (self.mamba_dim, self.trans_dim), dim=1)

        mamba_x = self.mamba_block(mamba_x)

        trans_x = Rearrange('b c h w -> b h w c')(trans_x)

        y = Rearrange('b c h w -> b h w c')(y)
        trans_x = self.trans_block(trans_x, y)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((mamba_x, trans_x), dim=1))
        x = x + res

        return x



class BMCBlock_dec(nn.Module):
    '''
    Binary Mamba-Conv Block for decoder
    '''
    def __init__(self, conv_dim, mamba_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(BMCBlock_dec, self).__init__()
        self.conv_dim = conv_dim
        self.mamba_dim = mamba_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.mamba_block = BiMambaBlock(
                hidden_dim=self.mamba_dim,
                drop_path=0,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=16,
                expand=2)
        self.conv1_1 = BConv2d(self.conv_dim+self.mamba_dim, self.conv_dim+self.mamba_dim, 1, bias=True)
        self.conv1_2 = BConv2d(self.conv_dim+self.mamba_dim, self.conv_dim+self.mamba_dim, 1, bias=True)

        self.conv_block = nn.Sequential(
                BConv2d(self.conv_dim, self.conv_dim, 3, bias=False),
                nn.ReLU(True),
                BConv2d(self.conv_dim, self.conv_dim, 3, bias=False)
                )

    def forward(self, x):
        conv_x, mamba_x = torch.split(self.conv1_1(x), (self.conv_dim, self.mamba_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        mamba_x = self.mamba_block(mamba_x) + mamba_x
        res = self.conv1_2(torch.cat((conv_x, mamba_x), dim=1))
        x = x + res
        return x

class UNetQBConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope=0.2):
        super(UNetQBConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = BinaryConv2dSkip1x1(in_size, out_size, 1)

        self.conv_1 = BinaryConv2dSkip1x1(in_size, out_size, kernel_size=3, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = BinaryConv2dSkip1x1(out_size, out_size, kernel_size=3, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = BConv2d(out_size, out_size , 1) 

        if downsample:
            self.downsample = nn.Conv2d(out_size, 2*out_size, 2, 2, 0, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
            
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    


class MainBMTNet(nn.Module):

    def __init__(self, in_nc=1, qb_channel=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256, fuse_before_downsample=True):
        super(MainBMTNet, self).__init__()
        self.k = (130*dim) / 64.0

        self.config = config
        self.dim = dim
        self.head_dim = int(dim/2)
        self.window_size = 8
        self.fuse_before_downsample = fuse_before_downsample

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        self.fourier_encoding = rff.layers.PositionalEncoding(sigma=1.0, m=10)
        self.intro_fourier_qb = HardBinaryConv(20*qb_channel, dim//2, kernel_size=3,
                              bias=True)
        self.intro_qb = nn.Conv2d(in_channels=qb_channel, out_channels=dim//2, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.qb_fusion = nn.Conv2d(in_channels=dim, out_channels=dim//2, kernel_size=1)
        self.depth = 3
        self.quad_encoders = nn.ModuleList()
        chan = dim//2
        begin = 0
        for i in range(self.depth):
            downsample = True if (i+1) < self.depth else False 
            # qb encoder
            if i < self.depth:
                self.quad_encoders.append(UNetQBConvBlock(chan, chan, downsample))
            chan = 2*chan
        self.m_down1 = nn.ModuleList([BMTBlock_enc(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [
                          nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)
                       ])

        begin += config[0]
        self.m_down2 = nn.ModuleList([BMTBlock_enc(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [
                          nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)
                      ])

        begin += config[1]
        self.m_down3 = nn.ModuleList([BMTBlock_enc(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [
                        nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)
                          ])

        begin += config[2]
        self.m_body = [BMCBlock_dec(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.Sequential(
                        nn.Conv2d(8*dim, 16*dim, 1, bias=False),
                        nn.PixelShuffle(2)
                        ),] + \
                      [BMCBlock_dec(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [nn.Sequential(
                        nn.Conv2d(4*dim, 8*dim, 1, bias=False),
                        nn.PixelShuffle(2)
                        ),] + \
                      [BMCBlock_dec(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [nn.Sequential(
                        nn.Conv2d(2*dim, 4*dim, 1, bias=False),
                        nn.PixelShuffle(2)
                        ),] + \
                    [BMCBlock_dec(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)  

    def forward(self, x0, qb_map):
        x0 = x0*self.k
        h, w = x0.size()[-2:]
        x0 = x0
        qb_map = qb_map
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        qb_map = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(qb_map)
        qb = self.fourier_encoding(rearrange(qb_map, 'b c h w -> b h w c'))
        qb = rearrange(qb, 'b h w c -> b c h w')
        qb = self.intro_fourier_qb(qb)
        qb2 = self.intro_qb(qb_map)
        qb = self.qb_fusion(torch.cat((qb,qb2), dim=1))
        qb_encs = []
        for i, down in enumerate(self.quad_encoders):
            if i < self.depth-1:
                qb, qb_up = down(qb, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    qb_encs.append(qb_up)
                else:
                    qb_encs.append(qb)
            else:
                qb = down(qb, self.fuse_before_downsample)
                qb_encs.append(qb)
                
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        
        x1 = self.m_head(x0)
        x2 = x1
        for i,  enc in enumerate(self.m_down1):
            if i < self.config[0]:
                x2 = enc(x2, qb_encs[0])
            else:
                x2 = enc(x2)
        x3 = x2
        for i,  enc in enumerate(self.m_down2):
            if i < self.config[1]:
                x3 = enc(x3, qb_encs[1])
            else:
                x3 = enc(x3)
        x4 = x3
        for i,  enc in enumerate(self.m_down3):
            if i < self.config[2]:
                x4 = enc(x4, qb_encs[2])
            else:
                x4 = enc(x4)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]/self.k
        
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

