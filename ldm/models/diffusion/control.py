import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import cv2
from einops import rearrange
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.models.diffusion.ddpm import DDPM
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim2 import DDIMSampler
from ldm.models.diffusion.plms2 import PLMSSampler
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from seecoder.lib.model_zoo.seecoder import SemanticContextEncoder
from ldm.modules.encoders.detail_encoder.encoder_acrossatt import dinov2_decoder
import numpy as np
def disabled_train(self, mode=True):
    return self

def randommask_loss(real_mask_size,num_patch=16, mask_rate=0.02):
    num_patches=num_patch * num_patch
    num_mask = max(1,int(mask_rate * num_patches))
    mask = torch.hstack([  # 水平方向叠起来
        torch.zeros(num_patches - num_mask),  # 25%为0
        torch.ones(num_mask),  # mask的部分设为1
    ])
    mask=mask.reshape((num_patch,num_patch))
    idx = torch.randperm(mask.nelement())
    mask = mask.view(-1)[idx].view(mask.size())
    mask_new=torch.zeros_like(real_mask_size).cuda()
    d=int(real_mask_size.shape[2]/num_patch)
    for i in range(num_patch):
        for j in range(num_patch):
            mask_new[:,:,i*d:i*d+d,j*d:j*d+d] = mask[i,j]
    return mask_new

# =============================================================
# 可训练部分 ControlNet
# =============================================================
class ControlNet(nn.Module):
    def __init__(
            self,
            in_channels,                                # 9
            model_channels,                             # 320
            hint_channels,                              # 20
            attention_resolutions,                      # [4,2,1]
            num_res_blocks,                             # 2
            channel_mult=(1, 2, 4, 8),                  # [1,2,4,4]
            num_head_channels=-1,                       # 64
            transformer_depth=1,                        # 1
            context_dim=None,                           # 768
            use_checkpoint=False,                       # True
            dropout=0,
            conv_resample=True,
            dims=2,
            num_heads=-1,
            use_scale_shift_norm=False):
        super(ControlNet, self).__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        # time 编码器
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # input 编码器
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # hint 编码器
        
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )


        # UNet
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    disabled_sa = False

                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
    
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
    
    def forward(self, x, hint, timesteps, reference_dino):
        # 处理输入
        context = reference_dino
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        
        guided_hint = self.input_hint_block(hint, emb, context)
       
        # 预测 control
        outs = []
        h = x.type(self.dtype)
        i=0
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            i=i+1
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                if i >= 3:
                    guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        
        return outs

# =============================================================
# 固定参数部分 ControlledUnetModel
# =============================================================
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None):
        hs = []
        
        # UNet 的上半部分
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        
        # 注入 control
        if control is not None:
            h += control.pop()
  
        # UNet 的下半部分
        for i, module in enumerate(self.output_blocks):
            
            h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        # 输出
        h = h.type(x.dtype)
        h = self.out(h)
        
        return h

# =============================================================
# 主干网络 ControlLDM
# =============================================================
class ControlLDM(DDPM):
    def __init__(self,
                 control_stage_config,          # ControlNet
                 first_stage_config,            # AutoencoderKL
                 cond_stage_config,             # FrozenCLIPImageEmbedder
                 scale_factor=1.0,              # 0.18215
                 *args, **kwargs):
        self.num_timesteps_cond = 1
        super().__init__(*args, **kwargs)                                       # self.model 和 self.register_buffer
        self.control_model = instantiate_from_config(control_stage_config)      # self.control_model
        self.instantiate_first_stage(first_stage_config)                        # self.first_stage_model 调用 AutoencoderKL
        self.instantiate_cond_stage(cond_stage_config)                          # self.cond_stage_model 调用 FrozenCLIPImageEmbedder
        #self.proj_out=nn.Linear(1024, 768)                 # 全连接层
        self.scale_factor = scale_factor                                        # 0.18215

        self.extraction = 'dinov2'
        self.t = 0
        if self.extraction=='dinov2':
            self.dinov2_vitl14 = torch.hub.load('E:/academic/Pytorch/crack-generate/self_DM_detail/dinov2', 'dinov2_vitl14',trust_repo=True, source='local')
            self.dinov2_vitl14.eval()
            self.dinov2_vitl14.train = disabled_train
            for param in self.dinov2_vitl14.parameters():
                param.requires_grad = False 
            self.linear = dinov2_decoder() #nn.Linear(1024, 768)
        
        if self.extraction=='seecoder':
            self.seecoder = SemanticContextEncoder()
            seecoder_dict = load_file('E:/academic/Pytorch/crack-generate/CAT-DM-main/seecoder/seecoder-control.safetensors', device='cpu')
            self.seecoder.load_state_dict(seecoder_dict, strict=True)
            self.seecoder.eval()
            self.seecoder.train = disabled_train
            for param in self.seecoder.parameters():
                param.requires_grad = False 
        self.unet_fozen()
        
    # AutoencoderKL 不训练
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    # FrozenCLIPImageEmbedder 不训练
    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        self.cond_stage_model.train = disabled_train
        self.cond_stage_model.image_encoder.requires_grad_(False)
        self.cond_stage_model.across.requires_grad_(True)
        
    def unet_fozen(self):
        self.model.to(self.device, self.dtype)
        self.model = self.model.eval()      
        for (name, param) in self.model.named_parameters():  
            if 'attn1' in name:
                param.requires_grad = True
            elif 'attn2' in name:
                param.requires_grad = True
           
        return self.model
    
    # 训练
    def training_step(self, batch, batch_idx):
        z_new, reference, hint, real_mask_size, pose, ref_pose= self.get_input(batch)               # 加载数据
        loss= self(z_new, reference, hint, real_mask_size, pose, ref_pose)                          # 计算损失
        self.log("loss",                                            # 记录损失
                 loss,                                  
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True)
        self.log('lr_abs',                                          # 记录学习率
                 self.optimizers().param_groups[0]['lr'], 
                 prog_bar=True, 
                 logger=True, 
                 on_step=True, 
                 on_epoch=False)
        return loss

    # 加载数据
    @torch.no_grad()
    def get_input(self, batch):
        
        # 加载原始数据
        x, inpaint, mask, reference, hint, real_mask, pose, ref_pose = super().get_input(batch)
        
        # AutoencoderKL 处理真值
        encoder_posterior = self.first_stage_model.encode(x)                           
        z = self.scale_factor * (encoder_posterior.sample()).detach()
        
        # AutoencoderKL 处理 inpaint
        encoder_posterior_inpaint = self.first_stage_model.encode(inpaint)     
        z_inpaint = self.scale_factor * (encoder_posterior_inpaint.sample()).detach()
        
        # Resize mask
        mask_resize = torchvision.transforms.Resize([z.shape[-2],z.shape[-1]],Image.BILINEAR)(mask)
        real_mask_size= torchvision.transforms.Resize([z.shape[-2],z.shape[-1]],Image.BILINEAR)(real_mask)
        mask_resize[mask_resize>=0.5]=1
        mask_resize[mask_resize<0.5]=0
        real_mask_size[real_mask_size>=0.5]=1
        real_mask_size[real_mask_size<0.5]=0
        
        # 整理 z_new
        z_new = torch.cat((z,z_inpaint,mask_resize),dim=1)
        
        out  = [z_new, reference, hint, real_mask_size, pose, ref_pose]
        
        return out
    
    # 计算损失
    def forward(self, z_new, reference, hint, real_mask_size, pose, ref_pose):
        self.t=self.t+1
        # 随机时间 t
        t = torch.randint(0, self.num_timesteps, (z_new.shape[0],), device=self.device).long()
        
        # CLIP 处理 reference
        reference_clip = self.cond_stage_model(reference,ref_pose,pose)
        
        if self.extraction=='CLIP':
            reference_dino = reference_clip
        elif self.extraction=='seecoder':
            reference_dino = self.seecoder(reference)
        else:
            # DINO 处理 reference
            dino = self.dinov2_vitl14(reference,is_training=True)
            dino1 = dino["x_norm_clstoken"].unsqueeze(1)
            dino2 = dino["x_norm_patchtokens"]
            reference_dino = torch.cat((dino1, dino2), dim=1)
            
            dino_pose = self.dinov2_vitl14(pose,is_training=True)
            dino_pose1 = dino_pose["x_norm_clstoken"].unsqueeze(1)
            dino_pose2 = dino_pose["x_norm_patchtokens"]
            pose = torch.cat((dino_pose1, dino_pose2), dim=1)
            
            dino_ref_pose = self.dinov2_vitl14(ref_pose,is_training=True)
            dino_ref_pose1 = dino_ref_pose["x_norm_clstoken"].unsqueeze(1)
            dino_ref_pose2 = dino_ref_pose["x_norm_patchtokens"]
            ref_pose = torch.cat((dino_ref_pose1, dino_ref_pose2), dim=1)
            
            reference_dino = self.linear(reference_dino,ref_pose,pose)
            
        # 随机加噪
        noise = torch.randn_like(z_new[:,:4,:,:])
        x_noisy = self.q_sample(x_start=z_new[:,:4,:,:], t=t, noise=noise)           
        x_noisy = torch.cat((x_noisy, z_new[:,4:,:,:]),dim=1)
        
        # 预测噪声
        model_output = self.apply_model(x_noisy, hint, t, reference_clip, reference_dino)
        
        # 计算损失
        
        loss1 = self.get_loss(model_output * real_mask_size, noise * real_mask_size, mean=False).mean([1, 2, 3])
        loss1 = self.get_loss(model_output , noise , mean=False).mean([1, 2, 3])
        
        mask = randommask_loss(real_mask_size,num_patch=16, mask_rate=0.5)
        loss2 = self.get_loss(model_output * mask, noise * mask, mean=True)
        loss = loss1.mean() + loss2
        
        return loss
    
    # 预测噪声
    def apply_model(self, x_noisy, hint, t, reference_clip, reference_dino):

        # 预测 control
        control = self.control_model(x_noisy, hint, t, reference_dino)

        # 调用 PBE
        model_output = self.model(x_noisy, t, reference_clip, control)

        return model_output
    
    # 优化器
    def configure_optimizers(self):
        # 学习率设置
        lr = self.learning_rate
        if self.extraction=='seecoder' or self.extraction=='CLIP':
            params = self.control_model.parameters()
        else:
            params = list(self.cond_stage_model.across.parameters()) + list(self.control_model.parameters())+list(self.linear.parameters())
            trainable_params, all_param = 0, 0
            for num in params:
                num_params = num.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(num, "ds_numel"):
                    num_params = num.ds_numel
                all_param += num_params
                if num.requires_grad:
                    trainable_params += num_params
            print('trainable_params, all_param',trainable_params, all_param)
        opt = torch.optim.AdamW(params, lr=lr)

        return opt
    
    # 采样
    @torch.no_grad()
    def sample_log(self, batch, ddim_steps=50, ddim_eta=0.):
        z_new, reference, hint, real_mask, pose, ref_pose = self.get_input(batch)
        x, _, mask, _, _, _,_,_ = super().get_input(batch)
        log = dict()

        # log["reference"] = reference
        # reconstruction = 1. / self.scale_factor * z_new[:,:4,:,:]
        # log["reconstruction"] = self.first_stage_model.decode(reconstruction)
        log["mask"] = mask

        test_model_kwargs = {}
        test_model_kwargs['inpaint_image'] = z_new[:,4:8,:,:]
        test_model_kwargs['inpaint_mask'] = z_new[:,8:,:,:]
        test_model_kwargs['input_image'] = z_new[:,:4,:,:] * (1-real_mask)
        test_model_kwargs['real_mask'] = real_mask
        test_model_kwargs['fg_img'] = z_new[:,0:4,:,:]
        test_model_kwargs['mask'] = real_mask
        ddim_sampler = DDIMSampler(self)
        #ddim_sampler = PLMSSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, _ = ddim_sampler.sample(ddim_steps, 
                                        reference.shape[0], 
                                        shape, 
                                        hint, 
                                        ref_pose,
                                        reference,
                                        pose,
                                        verbose=False, 
                                        eta=ddim_eta,
                                        x_T=None,
                                        test_model_kwargs=test_model_kwargs)
        samples = 1. / self.scale_factor * samples
        x_samples = self.first_stage_model.decode(samples[:,:4,:,:])
        # log["samples"] = x_samples

        x = torchvision.transforms.Resize([256, 256],Image.BILINEAR)(x)
        #reference = torchvision.transforms.Resize([512, 512],Image.BILINEAR)(hint[:,0:3,:,:])
        reference = torchvision.transforms.Resize([256, 256],Image.BILINEAR)(reference)
        x_samples = torchvision.transforms.Resize([256, 256],Image.BILINEAR)(x_samples)
        #log["grid"] = torch.cat((x, reference.repeat(1,3,1,1), x_samples), dim=2)
        log["grid"] = torch.cat((x, reference, x_samples), dim=2)
        
        return log
    


