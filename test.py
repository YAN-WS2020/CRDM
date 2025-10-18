import os
import torch
import argparse
import torchvision
import pytorch_lightning
import numpy as np

from PIL import Image
from torch import autocast
from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch.distributed as dist

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
def un_norm(x):
    return (x+1.0)/2.0

def un_norm_clip(x):
    x[0,:,:] = x[0,:,:] * 0.5 + 0.5
    x[1,:,:] = x[1,:,:] * 0.5 + 0.5
    x[2,:,:] = x[2,:,:] * 0.5 + 0.5
    return x

class DataModuleFromConfig(pytorch_lightning.LightningDataModule):
    def __init__(self,
                 batch_size,                       # 1
                 test=None,                        # {...}
                 wrap=False,                       # False
                 shuffle=False,             
                 shuffle_test_loader=False,
                 use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.wrap = wrap
        self.datasets = instantiate_from_config(test)
        self.dataloader = torch.utils.data.DataLoader(self.datasets, 
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_workers,
                                                      shuffle=shuffle,
                                                      worker_init_fn=None)



if __name__ == "__main__":
    # =============================================================
    # 处理 opt
    # =============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, default="configs/test_g.yaml")
    parser.add_argument("-c", "--ckpt", type=str, default=" ")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--ddim", type=int, default=100)
    opt = parser.parse_args()
    offset = [0, 0]
    # =============================================================
    # 设置 seed
    # =============================================================
    seed_everything(opt.seed)
    local_rank = 0 #int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5677'
    dist.init_process_group(backend='gloo', init_method='env://', rank = 0, world_size = 1)
    # =============================================================
    # 初始化 config
    # =============================================================
    config = OmegaConf.load(f"{opt.base}")

    # =============================================================
    # 加载 dataloader
    # =============================================================
    data = instantiate_from_config(config.data)
    print(f"{data.__class__.__name__}, {len(data.dataloader)}")

    # =============================================================
    # 加载 model
    # =============================================================
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.ckpt, map_location="cpu")["state_dict"], strict=False)
    model.cuda()
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model,extraction='dinov2') #'seecoder'，'CLIP'

    # =============================================================
    # 设置精度
    # =============================================================
    precision_scope = autocast

    # =============================================================
    # 开始测试
    # =============================================================
    dir = ""
    save_path = os.path.join(dir,"")
    save_path_img = os.path.join(save_path,'images/')
    save_path_direst = os.path.join(save_path,'direst/')
    save_path_mask = os.path.join(save_path,'masks/')

    
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)
    if not os.path.exists(save_path_direst):
        os.makedirs(save_path_direst)
    if not os.path.exists(save_path_mask):
        os.makedirs(save_path_mask)

    with torch.no_grad():
        with precision_scope("cuda"):
            for i,batch in enumerate(data.dataloader):
                # 加载数据
                inpaint = batch["inpaint_image"].to(torch.float32).to(device)
                reference = batch["ref_imgs"].to(torch.float32).to(device)
                mask = batch["inpaint_mask"].to(torch.float32).to(device)
                hint = batch["hint"].to(torch.float32).to(device)
                pose1 = batch["pose"].to(torch.float32).to(device)
                ref_pose = batch["ref_pose"].to(torch.float32).to(device)
                truth = batch["GT"].to(torch.float32).to(device)
                truth_mask = batch["mask"].to(torch.float32).to(device)
                real_mask = batch["real_mask"].to(torch.float32).to(device)
                fg_img = batch["fg_img"].to(torch.float32).to(device)
                bg_name = batch["bg_name"]
                fg_name = batch["fg_name"]
                # 数据处理
                encoder_img = model.first_stage_model.encode(truth)
                z = model.scale_factor * (encoder_img.sample()).detach()
                
                encoder_fg_img = model.first_stage_model.encode(fg_img)
                z_fg_img = model.scale_factor * (encoder_fg_img.sample()).detach()
                
                encoder_img_mask = model.first_stage_model.encode(truth)
     
                z_mask = model.scale_factor * (encoder_img_mask.sample()).detach()
                
                encoder_posterior_inpaint = model.first_stage_model.encode(inpaint)  # 潜空间转换
                # 添加噪声
                z_inpaint = model.scale_factor * (encoder_posterior_inpaint.sample()).detach()  # [1, 4, 64, 64]
                
                mask_resize = torchvision.transforms.Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(mask) # 默认是差值 bilinear
  
                real_mask_resize = torchvision.transforms.Resize((z_inpaint.shape[-2],z_inpaint.shape[-1]),Image.NEAREST)(real_mask) # 默认是差值 bilinear
                truth_mask_resize = torchvision.transforms.Resize((z_inpaint.shape[-2],z_inpaint.shape[-1]),Image.NEAREST)(truth_mask) # 默认是差值 bilinear
                
      
                real_mask_resize[real_mask_resize>=0.5]=1
                real_mask_resize[real_mask_resize<0.5]=0
                truth_mask_resize[truth_mask_resize>=0.5]=1
                truth_mask_resize[truth_mask_resize<0.5]=0
                
                
                test_model_kwargs = {}
                test_model_kwargs['image'] = z
                test_model_kwargs['input_image'] = z_mask
                test_model_kwargs['inpaint_image'] = z_inpaint
                test_model_kwargs['inpaint_mask'] = mask_resize
                test_model_kwargs['real_mask'] = real_mask_resize
                test_model_kwargs['mask'] = truth_mask_resize
                test_model_kwargs['fg_img'] = z_fg_img
                shape = (model.channels, model.image_size, model.image_size)
                
                # 预测结果
                samples, _ = sampler.sample(S=opt.ddim,
                                                 batch_size=1,
                                                 shape=shape,
                                                 pose=hint,
                                                 ref_pose=ref_pose,
                                                 conditioning=reference,
                                                 conditioning2=pose1,
                                                 verbose=False,
                                                 eta=0,
                                                 x_T= None, #z_mask, #None, #z_mask,
                                                 test_model_kwargs=test_model_kwargs,
                                                 unconditional_guidance_scale=1.0,
                                                 )
                samples = 1. / model.scale_factor * samples
                
                x_samples = model.first_stage_model.decode(samples[:,:4,:,:])

                x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                x_checked_image_torch = x_samples_ddim.cpu()
                truth = torch.clamp((truth + 1.0) / 2.0, min=0.0, max=1.0)
                fg_img = torch.clamp((fg_img + 1.0) / 2.0, min=0.0, max=1.0)
                
                x_checked_image_torch_C = x_checked_image_torch*real_mask.cpu() + truth.cpu()*(1-real_mask).cpu()
                x_checked_image_torch_mix = x_checked_image_torch*(1-truth_mask).cpu() + fg_img.cpu()*truth_mask.cpu()
                x_checked_image_torch_fusion = x_checked_image_torch*real_mask.cpu() + 0.5*truth.cpu()*(1-real_mask).cpu()+0.5*truth.cpu()*(1-real_mask).cpu()
                x_checked_image_torch_fusion_edge = x_checked_image_torch*(real_mask.cpu()-truth_mask.cpu()) + 0.5*truth.cpu()*(1-real_mask.cpu())+0.5*x_checked_image_torch*(1-real_mask.cpu()) + fg_img.cpu()*truth_mask.cpu()
                
                # 保存图像
                all_img=[]
                all_img_C = []

                truth_mask[truth_mask>=0.5]=1
                truth_mask[truth_mask<0.5]=0
                
                
                for j in range(x_checked_image_torch.shape[0]):

                    grid = x_checked_image_torch[j]
                    grid = 255. * grid.cpu().detach().numpy().transpose((1,2,0))
                    img = Image.fromarray(grid.astype(np.uint8))
                    img.save(save_path_direst+fg_name[j].split('-')[-1].split('.')[0]+"-"+bg_name[j].replace('jpg','png'))

                    grid_C = x_checked_image_torch_C[j]
                    grid_C = 255. * grid_C.cpu().detach().numpy().transpose((1,2,0))
                    img_C = Image.fromarray(grid_C.astype(np.uint8))
                    img_C.save(save_path_img+fg_name[j].split('-')[-1].split('.')[0]+"-"+bg_name[j].replace('jpg','png'))
                    
                    mask = truth_mask
                    mask = 255. * mask[j][0].cpu().detach().numpy()
                    img = Image.fromarray(mask.astype(np.uint8))
                    img.save(save_path_mask+fg_name[j].split('-')[-1].split('.')[0]+"-"+bg_name[j].replace('jpg','png'))
                    