from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import os
from ldm.modules.diffusionmodules.util import checkpoint
import numpy as np
import cv2
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

   
class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.t=0

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        self.t=self.t+1
        x1 = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) #+ x
        #if x1.shape[1]==1024:
            #if self.t>=int(self.t/101)*100+95:
                #self.clip_features_to_heatmap(features=x1, t=self.t)
        x = x1 + x
        
        x2 = self.attn2(self.norm2(x), context=context)
        
        x= x2 + x

        x = self.ff(self.norm3(x)) + x
        
        #print('attention',x.shape)
        
        return x
    def clip_features_to_heatmap(self, features, t, method='mean', target_size=None, colormap=cv2.COLORMAP_JET):
        
        save_dir='E:/academic/Pytorch/crack-generate/self_DM_detail/logs/CRDM-pose+img-imageloss/test/attention2/img+refpose/'
        existing_files = [f for f in os.listdir(save_dir) if f.startswith(f'attention_{t}')]
        num = len(existing_files)+1
        
        
        """
        将CLIP特征张量转换为热力图
        
        参数:
        - features: CLIP输出的特征张量 (3084, 1024)
        - original_image: 原始输入图像
        - method: 特征聚合方法 ('mean', 'max', 'pca')
        - target_size: 目标热力图尺寸
        """
        
        # 确保特征张量是二维的
        #assert len(features.shape) == 3, "特征张量应该是二维的"
        #assert features.shape[1] == 3084, "特征数量应该是3084"
        #assert features.shape[2] == 1024, "特征维度应该是1024"
        
        # 将特征转换为numpy数组
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        features=np.mean(features, axis=0)
        # 方法1: 均值聚合
        if method == 'mean':
            # 对1024个特征维度求均值，得到3084个空间位置的重要性分数
            importance_scores = np.mean(features, axis=1)
        
        # 方法2: 最大值聚合
        elif method == 'max':
            importance_scores = np.max(features, axis=1)
        
        # 方法3: PCA主成分分析
        elif method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            importance_scores = pca.fit_transform(features).flatten()
        
        else:
            raise ValueError("不支持的聚合方法，请选择 'mean', 'max' 或 'pca'")
        
        # 重塑为特征图的空间尺寸 (假设是77x40的网格，77*40=3080，可能有4个额外的位置)
        # CLIP Vision Transformer通常使用patch大小为16，图像尺寸为224x224时产生14x14=196个patch
        # 3084可能是其他配置，我们需要找到合适的网格尺寸
        
        # 寻找最接近的网格尺寸
        grid_sizes = []
        for h in range(1, int(np.sqrt(features.shape[0])) + 1):
            if features.shape[0] % h == 0:
                grid_sizes.append((h, features.shape[0] // h))
        
        if grid_sizes:
            # 选择最接近正方形的网格
            height, width = max(grid_sizes, key=lambda x: min(x[0]/x[1], x[1]/x[0]))
        else:
            # 如果不能整除，使用近似尺寸
            height, width = 56, 56  # 42*74=3108，略大于3084
        
        height, width = int(np.sqrt(features.shape[0])), int(np.sqrt(features.shape[0]))
        # 创建热力图
        heatmap = np.zeros(height * width)
        
        heatmap[:len(importance_scores)] = importance_scores
        heatmap = heatmap.reshape(height, width)
        #print(heatmap.shape)
        # 调整热力图尺寸以匹配原始图像
        #print(heatmap.shape)
        target_size = (224, 224)
        
        heatmap = cv2.resize(heatmap, target_size,interpolation=cv2.INTER_NEAREST)
        attention_path=os.path.join('E:/academic/Pytorch/crack-generate/self_DM_detail/logs/CRDM-pose+img-imageloss/test/attention2/img+refpose/',f'{t}_attention_numpy.npy')
        if num==1:
            attention_numpy=heatmap
            attention_numpy=np.expand_dims(attention_numpy, axis=0)
        else:
            attention_numpy = np.load(attention_path)
            attention_numpy=np.concatenate((attention_numpy,np.expand_dims(heatmap, axis=0)), axis=0)

        np.save(os.path.join(attention_path),attention_numpy)
 
        
        # 归一化热力图
        heatmap_resized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'E:/academic/Pytorch/crack-generate/self_DM_detail/logs/CRDM-pose+img-imageloss/test/attention2/img+refpose/attention_{t}_1_{num}.png', heatmap_colored )

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.t=0

    def forward(self, x, context=None):
        
        self.t=self.t+1
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in