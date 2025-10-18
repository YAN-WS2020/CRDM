from typing import List
import torch
from torchvision import transforms
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModel as OriginalCLIPVisionModel
from ._clip import CLIPVisionModel
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os
from einops import rearrange
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import cv2
import numpy as np
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
if is_torch2_available():
    from .attention_processor import SSRAttnProcessor2_0 as SSRAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .attention_processor import SSRAttnProcessor, AttnProcessor
from .resampler import Resampler


def clip_features_to_heatmap(features, method='mean', target_size=None,colormap=cv2.COLORMAP_JET):
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
    features=features.squeeze(0)
    # 将特征转换为numpy数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
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
    for h in range(1, int(np.sqrt(3084)) + 1):
        if 3084 % h == 0:
            grid_sizes.append((h, 3084 // h))
    
    if grid_sizes:
        # 选择最接近正方形的网格
        height, width = max(grid_sizes, key=lambda x: min(x[0]/x[1], x[1]/x[0]))
    else:
        # 如果不能整除，使用近似尺寸
        height, width = 56, 56  # 42*74=3108，略大于3084
    
    height, width = 56, 56
    # 创建热力图
    heatmap = np.zeros(height * width)
    
    heatmap[:len(importance_scores)] = importance_scores
    heatmap = heatmap.reshape(height, width)
    
    # 调整热力图尺寸以匹配原始图像
    print(heatmap.shape)
    target_size = (224, 224)
    
    heatmap = cv2.resize(heatmap, target_size,interpolation=cv2.INTER_NEAREST)
    
    # 归一化热力图
    heatmap_resized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'E:/academic/Pytorch/crack-generate/self_DM_detail/logs/CRDM-pose+img-0.5loss/test/attention/1.jpg', heatmap_colored )

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        #self.norm = torch.nn.LayerNorm(cross_attention_dim)
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        #clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
 
        #self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
 
        self.Wq = nn.Linear(in_channels, emb_dim)
        self.Wk = nn.Linear(in_channels, emb_dim)
        self.Wv = nn.Linear(in_channels, emb_dim)
 
        self.proj_out = nn.Linear(emb_dim, 768)#nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm = torch.nn.LayerNorm(in_channels)
        #self.norm2 = torch.nn.LayerNorm(768)
 
    def forward(self, img, ref_pose, pose, pad_mask=None):
        '''
        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        #print(x.shape)
        #b, c, h, w = x.shape
 
        #x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        #x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
        img_normal = self.norm(img)
        ref_pose_normal = self.norm(ref_pose)
        pose_normal = self.norm(pose)
 
        Q = self.Wq(ref_pose_normal)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(pose_normal)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(img_normal)
 
        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale
 
        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]
        
        #out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = out +  pose_normal + img_normal # + pose_normal + #
        out = self.proj_out(out)   # [batch_size, c, h, w] 
        #out = self.norm2(out)
        #print(out.shape)
        return out


class detail_encoder(torch.nn.Module):
    """from SSR-encoder"""
    def __init__(self, image_encoder_path, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # load image encoder
        clip_encoder = OriginalCLIPVisionModel.from_pretrained(image_encoder_path)
        self.image_encoder = CLIPVisionModel(clip_encoder.config)
        state_dict = clip_encoder.state_dict()
        self.image_encoder.load_state_dict(state_dict, strict=False)
        self.image_encoder.to(self.device, self.dtype)
        del clip_encoder
        self.clip_image_processor = CLIPImageProcessor()
        # load SSR layers
        self.across = CrossAttention(in_channels=1024, emb_dim=1024, att_dropout=0.0, aropout=0.0)

    def forward(self, img,ref_pose,pose):
       
        image_embeds = self.image_encoder(img, output_hidden_states=True)['hidden_states'][2::2]
        image_embeds = torch.cat(image_embeds, dim=1)
        
        
        ref_pose_embeds = self.image_encoder(ref_pose, output_hidden_states=True)['hidden_states'][2::2]
        ref_pose_embeds = torch.cat(ref_pose_embeds, dim=1)

       
        pose_embeds = self.image_encoder(pose, output_hidden_states=True)['hidden_states'][2::2]
        pose_embeds = torch.cat(pose_embeds, dim=1)
        
        
        embeds = self.across(image_embeds,ref_pose_embeds,pose_embeds)
       
        return embeds
    '''
    def forward(self, ref_img,ref_pose,img):
        ref_image_embeds = self.image_encoder(ref_img, output_hidden_states=True)['hidden_states'][2::2]
        ref_image_embeds = torch.cat(ref_image_embeds, dim=1)
        
        ref_pose_embeds = self.image_encoder(ref_pose, output_hidden_states=True)['hidden_states'][2::2]
        ref_pose_embeds = torch.cat(ref_pose_embeds, dim=1)
        ref_pose_encoder = self.resampler(ref_pose_embeds)
        
        embeds = self.across(ref_image_embeds,ref_pose_embeds)
       
        embeds = embeds + ref_pose_encoder
        
        image_embeds = self.image_encoder(img, output_hidden_states=True)['hidden_states'][2::2]
        image_embeds = torch.cat(image_embeds, dim=1)
        image_embeds = self.across(image_embeds,image_embeds)
        
        #print(embeds.shape)
        return embeds, image_embeds
    '''
    @torch.inference_mode()
    def get_image_embeds(self, img,ref_pose,pose):
        
        image_embeds = self.image_encoder(img, output_hidden_states=True)['hidden_states'][2::2]
        image_embeds = torch.cat(image_embeds, dim=1)
        #ref_pose_embeds = torch.zeros_like(ref_pose_embeds)
        #image_embeds = self.image_encoder(img).last_hidden_state
        
        ref_pose_embeds = self.image_encoder(ref_pose, output_hidden_states=True)['hidden_states'][2::2]
        ref_pose_embeds = torch.cat(ref_pose_embeds, dim=1)
        
        #ref_pose_encoder = self.resampler(ref_pose_embeds)
        #ref_pose_embeds = self.image_encoder(ref_pose).last_hidden_state
        
        pose_embeds = self.image_encoder(pose, output_hidden_states=True)['hidden_states'][2::2]
        pose_embeds = torch.cat(pose_embeds, dim=1)
        #pose_embeds = self.image_encoder(pose).last_hidden_state
        
        clip_image_embeds = self.across(image_embeds,ref_pose_embeds,pose_embeds)
        #clip_image_embeds= self.resampler(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(img), output_hidden_states=True)['hidden_states'][2::2]
        uncond_clip_image_embeds = torch.cat(uncond_clip_image_embeds, dim=1)
        #uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(img)).last_hidden_state
        
        uncond_clip_image_embeds = self.across(uncond_clip_image_embeds,uncond_clip_image_embeds,uncond_clip_image_embeds)
        
        return clip_image_embeds, uncond_clip_image_embeds
    

class dinov2_decoder(torch.nn.Module):
    """from SSR-encoder"""
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # load SSR layers
        self.resampler = self.init_proj()
        self.across = CrossAttention(in_channels=1024, emb_dim=1024, att_dropout=0.0, aropout=0.0)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=768,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=1,
        ).to(self.device, dtype=torch.float32)
        return image_proj_model

    def forward(self, image_embeds,ref_pose_embeds,pose_embeds):

        embeds = self.across(image_embeds,ref_pose_embeds,pose_embeds)
        
        return embeds

    @torch.inference_mode()
    def get_image_embeds(self, image_embeds,ref_pose_embeds, pose_embeds,zeros_embeds):

        # cond
        #image_embeds = self.resampler(image_embeds)
        #pose_embeds = self.resampler(pose_embeds)
        clip_image_embeds = self.across(image_embeds,ref_pose_embeds,pose_embeds)
        
        uncond_clip_image_embeds = self.across(zeros_embeds,zeros_embeds,zeros_embeds)
        
        return clip_image_embeds, uncond_clip_image_embeds
