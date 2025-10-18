import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as tf
from PIL import Image
from pathlib import Path
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2
from math import *
import math
import random


class TestDataset(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired"):
        self.state=state
        self.dataset_dir = dataset_dir
        self.type = type
        self.back_file = os.path.join(dataset_dir,self.state,"back")
        self.back_list=[path.name for path in Path(self.back_file).glob('*.*')]
        #self.back_list = random.sample(self.back_list,350)
        if type == "paired":
            self.fg_img_file = os.path.join(dataset_dir,self.state,"fg_img")
            self.fg_img_list=[path.name for path in Path(self.fg_img_file).glob('*.*')]
            self.ref_img_file = os.path.join(dataset_dir,self.state, "ref_img")
            self.ref_img_list=[path.name for path in Path(self.ref_img_file).glob('*.*')]
           

    def __len__(self):
        return len(self.ref_img_list)
    
    def __getitem__(self, index):

        image_name = self.back_list[index]
        # 确定路径
        img_path = os.path.join(self.back_file, image_name)
        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.Resize((256,256),Image.BILINEAR)(img)
        
        ref_name = self.ref_img_list[index]
        # 确定路径
        ref_path = os.path.join(self.ref_img_file, ref_name)
        ref_img = Image.open(ref_path).convert("RGB")
        
        ref_mask_path = os.path.join(self.ref_img_file.replace('ref_img','ref_mask'), ref_name.replace('jpg','png'))
        ref_mask = Image.open(ref_mask_path).convert("L")
        
        ref_pose = img
        
        fg_name = self.fg_img_list[index] #np.random.choice(self.fg_img_list)
        fg_img_path = os.path.join(self.fg_img_file, fg_name)
        fg_img = Image.open(fg_img_path).convert("RGB")
        
        mask_path = os.path.join(self.fg_img_file.replace('fg_img','fg_mask'), fg_name.replace('jpg','png'))
        mask = Image.open(mask_path).convert("L")
        
        
        mask_gt = torchvision.transforms.ToTensor()(mask)
        ref_mask = torchvision.transforms.ToTensor()(ref_mask)
        
        max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        max_mask = max_pool(mask_gt)
        max_mask = max_pool(max_mask)
        max_mask[max_mask>=0.5]=1
        max_mask[max_mask<0.5]=0
        
        ref_max_mask= max_pool(ref_mask)
        ref_max_mask= max_pool(ref_max_mask)
        ref_max_mask[ref_max_mask>=0.5]=1
        ref_max_mask[ref_max_mask<0.5]=0
        
        max_mask=Image.fromarray(max_mask.squeeze().numpy())
        ref_max_mask=Image.fromarray(ref_max_mask.squeeze().numpy())
        ref_mask=Image.fromarray(ref_mask.squeeze().numpy())
 
        ref_aug, ref_mask2, ref_max_mask2 = ref_img, ref_mask, ref_max_mask
        
        ref_aug = torchvision.transforms.ToTensor()(ref_aug)
        ref_max_mask2 = torchvision.transforms.ToTensor()(ref_max_mask2)
        ref_max_mask2_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(ref_max_mask2).squeeze(0)
        ref_max_mask2_224[ref_max_mask2_224>=0.5]=1
        ref_max_mask2_224[ref_max_mask2_224<0.5]=0
        
        ref_mask2 = torchvision.transforms.ToTensor()(ref_mask2)
        ref_mask2_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(ref_mask2).squeeze(0)
        ref_mask2_224[ref_mask2_224>=0.4]=1
        ref_mask2_224[ref_mask2_224<0.4]=0
        
        ref_edge_224 = ref_max_mask2_224-ref_mask2_224
        
        ref_pose_224=torch.zeros(3,224,224)
        
        ref_pose_224[0,:,:][ref_edge_224==1]=20/255
        ref_pose_224[1,:,:][ref_edge_224==1]=80/255
        ref_pose_224[2,:,:][ref_edge_224==1]=194/255
        
        ref_pose_224[0,:,:][ref_mask2_224==1]=248/255
        ref_pose_224[1,:,:][ref_mask2_224==1]=251/255
        ref_pose_224[2,:,:][ref_mask2_224==1]=14/255
         
        img1 = torchvision.transforms.ToTensor()(img)
        fg_img1 = torchvision.transforms.ToTensor()(fg_img)
        mask1 = torchvision.transforms.ToTensor()(mask)
        mask1[mask1>=0.5]=1
        mask1[mask1<0.5]=0
        max_mask1 = torchvision.transforms.ToTensor()(max_mask)
        max_mask1[max_mask1>=0.5]=1
        max_mask1[max_mask1<0.5]=0
        
        fg_mask_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(mask1).squeeze(0)
        fg_mask_224[fg_mask_224>=0.4]=1
        fg_mask_224[fg_mask_224<0.4]=0
        
        fg_max_mask1_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(max_mask1).squeeze(0)
        fg_max_mask1_224[fg_max_mask1_224>=0.4]=1
        fg_max_mask1_224[fg_max_mask1_224<0.4]=0
        
        fg_edge_224 = fg_max_mask1_224-fg_mask_224
        
        fg_pose_224=torch.zeros(3,224,224)
        
        fg_pose_224[0,:,:][fg_edge_224==1]=20/255
        fg_pose_224[1,:,:][fg_edge_224==1]=80/255
        fg_pose_224[2,:,:][fg_edge_224==1]=194/255
        
        fg_pose_224[0,:,:][fg_mask_224==1]=248/255
        fg_pose_224[1,:,:][fg_mask_224==1]=251/255
        fg_pose_224[2,:,:][fg_mask_224==1]=14/255
        
        ref_img = torchvision.transforms.ToTensor()(ref_img)
        
        edge = max_mask1-mask1
        
        ref = ref_aug * ref_max_mask2 + (1-ref_max_mask2)
        refernce = torchvision.transforms.Resize((224,224),Image.BILINEAR)(ref)
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))(refernce)
        
        img1 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img1)
        fg_img1 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(fg_img1)
        # 生成 inpaint 和 hint
        inpaint = img1 * (1-max_mask1) #img1 * (1-max_mask1) + edge * torch.randn_like(img1)
        
        hint1 = torch.cat((mask1,edge),dim = 0)

        return {"GT": img1,                  # [3, 512, 512]
                "inpaint_image": inpaint,   # [3, 512, 512]
                "inpaint_mask": 1-max_mask1,   # [1, 512, 512]
                "ref_imgs": refernce,       # [3, 224, 224]
                "hint": hint1,           # [6, 512, 512]
                "bg_name": image_name,
                "fg_name": fg_name,
                "ref_name": ref_name,
                "real_mask": max_mask1,
                "mask": mask1,
                "fg_img": fg_img1,
                "pose":fg_pose_224,
                "ref_pose":ref_pose_224
                }
