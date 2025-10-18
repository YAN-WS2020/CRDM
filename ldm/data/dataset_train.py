import os
import torch
import torchvision
import torch.utils.data as data

import torchvision.transforms.functional as tf
from PIL import Image
from pathlib import Path
import pandas as pd
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2
from math import *
import math
import random


class Augmentations_PIL:
    def __init__(self, augment_mode, input_shape):
        self.augment_dict=augment_mode
        self.image_fill = 255  # image fill=0，0对应黑边
        self.label_fill = 0  # label fill=0，0对应黑边
        self.input_hw=input_shape
    
    def rotation128(self, image, pose, label, max_mask):
        angle = self.augment_dict["rotation128"]["angle"]
        image = tf.rotate(image, angle, expand=True,fill=self.image_fill)
        pose = tf.rotate(pose, angle, expand=True,fill=(0,0,0))
        label = tf.rotate(label, angle, expand=True,fill=self.label_fill)
        max_mask = tf.rotate(max_mask, angle, expand=True,fill=self.label_fill)
        image = tf.resize(image,self.input_hw)
        pose = tf.resize(pose,self.input_hw,transforms.InterpolationMode.NEAREST)
        label = tf.resize(label,self.input_hw,transforms.InterpolationMode.NEAREST)
        max_mask = tf.resize(max_mask,self.input_hw,transforms.InterpolationMode.NEAREST)
        return image, pose, label, max_mask
    
    def rotation_crop(self, image, pose, label, max_mask):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :param angle:  None, list-float, tuple-float
        :return:  PIL
        '''
        angle1 = self.augment_dict["rotation_crop"]["angle"]
        angle = transforms.RandomRotation.get_params(angle1)
    
        h,w=image.size[1],image.size[0]
        image = tf.rotate(image, angle, expand=True,fill=255)
        pose = tf.rotate(pose, angle, expand=True,fill=0)
        label = tf.rotate(label, angle, expand=True,fill=0)
        max_mask = tf.rotate(max_mask, angle, expand=True,fill=0)
        alpha=math.atan(h/w)

        crop_W=int(h*fabs(cos(alpha)))
        crop_H=int(h*fabs(sin(alpha)))
        #H=int(h*fabs(cos(alpha))+w*fabs(sin(alpha)))
        #W=int(h*fabs(sin(alpha))+w*fabs(cos(alpha)))
        H=image.size[1]
        W=image.size[0]
        top=int(H*0.5-crop_H*0.5)
        left=int(W*0.5-crop_W*0.5)

        image=tf.crop(image,top,left,crop_H,crop_W)
        pose=tf.crop(pose,top,left,crop_H,crop_W)
        label=tf.crop(label,top,left,crop_H,crop_W)
        max_mask=tf.crop(max_mask,top,left,crop_H,crop_W)

        image = tf.resize(image,(h,w))
        pose = tf.resize(pose,(h,w),transforms.InterpolationMode.NEAREST)
        label = tf.resize(label,(h,w),transforms.InterpolationMode.NEAREST)
        max_mask = tf.resize(max_mask,(h,w),transforms.InterpolationMode.NEAREST)
        return image, pose, label, max_mask
        

    def flipH(self, image, pose, label, max_mask):
        image = tf.hflip(image)  # 水平翻转
        pose = tf.hflip(pose)
        label = tf.hflip(label)
        max_mask = tf.hflip(max_mask)
        return image, pose, label, max_mask

    def flipV(self,image,pose, label, max_mask):
        image = tf.vflip(image)  # 垂直翻转
        pose = tf.vflip(pose)
        label = tf.vflip(label)
        max_mask = tf.vflip(max_mask)
        return image, pose, label, max_mask

    # gassian noise
    def gaussianblur(self, image, pose, label, max_mask):
        sigma = self.augment_dict["gaussianblur"]["sigma"]
        kernel_size=np.random.choice([3,5,7])
        transforms_func = transforms.GaussianBlur(kernel_size, sigma)
        image = transforms_func(image)

        return image, pose, label, max_mask

    
    def rescale(self,image,pose,label, max_mask):
        ratio= self.augment_dict["rescale"]["rescale_ratio"]
        h,w=self.input_hw 
        w_src,h_src=image.size
        image=np.array(image)
        pose=np.array(pose)
        label=np.array(label)
        max_mask = np.array(max_mask)
        max_reshape_ratio = min(h / h_src, w / w_src)
        rescale_ratio = np.random.uniform(ratio, max_reshape_ratio)
        # reshape src img and mask
        rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
        image_rescale = cv2.resize(image, (rescale_w, rescale_h),
                            interpolation=cv2.INTER_LINEAR)
        pose = cv2.resize(pose, (rescale_w, rescale_h),
                            interpolation=cv2.INTER_NEAREST)
        mask_rescale = cv2.resize(label, (rescale_w, rescale_h),
                            interpolation=cv2.INTER_NEAREST)
        max_mask_rescale = cv2.resize(max_mask, (rescale_w, rescale_h),
                            interpolation=cv2.INTER_NEAREST)
        
        image_filled = cv2.copyMakeBorder(image_rescale, (h - rescale_h), (h - rescale_h), (w - rescale_w), (w - rescale_w),  borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        pose = cv2.copyMakeBorder(pose, (h - rescale_h), (h - rescale_h), (w - rescale_w), (w - rescale_w),  borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
        mask_filled = cv2.copyMakeBorder(mask_rescale, (h - rescale_h), (h - rescale_h), (w - rescale_w), (w - rescale_w),  borderType=cv2.BORDER_CONSTANT,value=0)
        max_mask_filled = cv2.copyMakeBorder(max_mask_rescale, (h - rescale_h), (h - rescale_h), (w - rescale_w), (w - rescale_w),  borderType=cv2.BORDER_CONSTANT,value=0)
        
        # set paste coord
        py = int(np.random.random() * (h - rescale_h))
        px = int(np.random.random() * (w - rescale_w))
    
        # paste src img and mask to a zeros background
        img_new   = np.zeros((h, w, 3), dtype=np.uint8)
        pose_new   = np.zeros((h, w, 3), dtype=np.uint8)
        mask_new  = np.zeros((h, w), dtype=np.uint8)
        max_mask_new  = np.zeros((h, w), dtype=np.uint8)
        img_new   = image_filled[py:(py+h),px:(px+w),:]
        pose_new   = pose[py:(py+h),px:(px+w),:]
        mask_new  = mask_filled[py:(py+h),px:(px+w)]
        max_mask_new  = max_mask_filled[py:(py+h),px:(px+w)]
        
        img_new = Image.fromarray(img_new)
        pose_new = Image.fromarray(pose_new)
        mask_new = Image.fromarray(mask_new)
        max_mask_new = Image.fromarray(max_mask_new)
        return img_new, pose_new, mask_new, max_mask_new
    
    def random_crop(self,image,pose,label,max_mask):
        ratio= self.augment_dict["random_crop"]["ratio"]
        crop_ratio = np.random.uniform(1, ratio)
        h,w=self.input_hw 
        w_src,h_src=image.size
        image=np.array(image)
        pose=np.array(pose)
        label=np.array(label)
        max_mask=np.array(max_mask)
        box_h,box_w= int(h / crop_ratio),int(w / crop_ratio)
        points=np.where(label[int(box_h/2):int(h_src-box_h/2),int(box_w/2):int(w_src-box_w/2)]==255)
        if len(points[0])==0:
            center_x=np.random.randint(int(box_h/2),int(h_src-box_h/2))
            center_y=np.random.randint(int(box_w/2),int(w_src-box_w/2))
        else:
            i=np.random.choice(len(points[0]))
            center_x=points[0][i]+int(box_h/2)
            center_y=points[1][i]+int(box_w/2)
        
        
        image_new=image[center_x-int(box_h/2):center_x+int(box_h/2),center_y-int(box_w/2):center_y+int(box_w/2),:]
        pose=pose[center_x-int(box_h/2):center_x+int(box_h/2),center_y-int(box_w/2):center_y+int(box_w/2),:]
        mask_new=label[center_x-int(box_h/2):center_x+int(box_h/2),center_y-int(box_w/2):center_y+int(box_w/2)]
        max_mask_new=max_mask[center_x-int(box_h/2):center_x+int(box_h/2),center_y-int(box_w/2):center_y+int(box_w/2)]
        
        image_new = cv2.resize(image_new, (w, h),
                            interpolation=cv2.INTER_LINEAR)
        pose = cv2.resize(pose, (w, h),
                            interpolation=cv2.INTER_NEAREST)
        mask_new = cv2.resize(mask_new, (w, h),
                            interpolation=cv2.INTER_NEAREST)
        max_mask_new = cv2.resize(max_mask_new, (w, h),
                            interpolation=cv2.INTER_NEAREST)
        
        img_new = Image.fromarray(image_new)
        pose = Image.fromarray(pose)
        mask_new = Image.fromarray(mask_new)
        max_mask_new = Image.fromarray(max_mask_new)
        return img_new, pose, mask_new, max_mask_new
    
    def translate(self,image,pose,label, max_mask):
        w_rate,h_rate= self.augment_dict["translate"]["w_rate"],self.augment_dict["translate"]["h_rate"]
        image_compose=np.array(image)
        pose=np.array(pose)
        label_compose=np.array(label)
        max_mask_compose=np.array(max_mask)
        x=int(self.input_hw[1]*(w_rate-0.125))
        y=int(self.input_hw[0]*(h_rate-0.125))
        M=np.float32([[1,0,x],[0,1,y]])
        image_compose=cv2.warpAffine(image_compose,M,(self.input_hw[1],self.input_hw[0]),borderValue=(255,255,255))#borderMode=cv2.BORDER_REPLICATE,
        pose=cv2.warpAffine(pose,M,(self.input_hw[1],self.input_hw[0]),borderValue=(0,0,0))#borderMode=cv2.BORDER_REPLICATE,
        label_compose=cv2.warpAffine(label_compose,M,(self.input_hw[1],self.input_hw[0]),borderValue=0)
        max_mask_compose=cv2.warpAffine(max_mask_compose,M,(self.input_hw[1],self.input_hw[0]),borderValue=0)
        image_compose = Image.fromarray(image_compose)
        pose = Image.fromarray(pose)
        label_compose = Image.fromarray(label_compose)
        max_mask_compose = Image.fromarray(max_mask_compose)
        return image_compose, pose, label_compose, max_mask_compose
    
    
class Transforms_PIL(object):
    def __init__(self, augment_list, input_shape):
        self.aug_pil = Augmentations_PIL(augment_list,input_shape)
        self.augment_ways=augment_list
        self.aug_funcs = [a for a in self.aug_pil.__dir__() if not a.startswith('_') and a not in self.aug_pil.__dict__]

    def __call__(self, image, pose, label, max_mask):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :return:  PIL
        '''
        name=np.random.choice(list(self.augment_ways.keys()))
        image, pose, label, max_mask = getattr(self.aug_pil, str(name))(image, pose, label, max_mask)
        return image, pose, label, max_mask

    
def Basic():
    augment_mode = {}
    
    tem_1 = {}
    tem_1["angle"] = np.random.randint(-20,20)
    augment_mode["rotation128"] = tem_1
    
    tem_2 = {}
    tem_2["angle"] = (-20,20)
    augment_mode["rotation_crop"] = tem_2
    
    augment_mode["flipH"] = True
    
    augment_mode["flipV"] = True
    
    tem_5 = {}
    tem_5["distortion_scale"] = random.uniform(0.,0.3)
    augment_mode["perspective"] = tem_5
    
    tem_6 = {}
    tem_6["sigma"] = 5
    augment_mode["gaussianblur"] = tem_6
    
    tem_7 = {}
    tem_7["rescale_ratio"] = 0.8
    augment_mode["rescale"] = tem_7
    
    tem_9 = {}
    tem_9["w_rate"] = np.random.uniform(0,0.25)
    tem_9["h_rate"] = np.random.uniform(0,0.25)
    augment_mode["translate"] = tem_9

    return augment_mode

def Basic2():
    augment_mode = {}
    
    augment_mode['self_images']=True
    
    tem_1 = {}
    tem_1["angle"] = 180
    augment_mode["rotation128"] = tem_1
    
    augment_mode["flipH"] = True
    
    augment_mode["flipV"] = True
    
    
    return augment_mode





class ImageDataset(data.Dataset):
    def __init__(self, state, dataset_dir,augment_img=True,augment_fg=True,):
        self.state=state
        self.dataset_dir = dataset_dir
        self.dataset_list = []
        self.augment_img=augment_img
        self.augment_fg=augment_fg

        if state == "train":
            self.dataset_file = os.path.join(dataset_dir,self.state,"images")
            self.dataset_list=[path.name for path in Path(self.dataset_file).glob('*.jpg')]

    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, index):
        
        image_name = self.dataset_list[index]

        # 确定路径
        img_path = os.path.join(self.dataset_dir, self.state, "images", image_name)
        mask_path = os.path.join(self.dataset_dir, self.state, "masks", image_name.replace('.jpg','.png'))                       
        
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        pose = img
        mask = Image.open(mask_path).convert("L")
        
        mask_gt = torchvision.transforms.ToTensor()(mask)
        
        max_pool = nn.MaxPool2d(kernel_size=15, stride=1, padding=7)
        max_mask = max_pool(mask_gt)
        max_mask = max_pool(max_mask)
        max_mask[max_mask>=0.5]=1
        max_mask[max_mask<0.5]=0
        max_mask=Image.fromarray(max_mask.squeeze().numpy())
        if self.augment_img:
            aug = Transforms_PIL(Basic2(), (256,256))
            img1, pose1, mask1, max_mask1 = aug(img, pose, mask, max_mask)
        else:
            img1, pose1, mask1, max_mask1 = img, pose, mask, max_mask
        
        img_512 = torchvision.transforms.ToTensor()(img1)
        
        max_mask256 = torchvision.transforms.ToTensor()(max_mask1)
        max_mask256[max_mask256>=0.5]=1
        max_mask256[max_mask256<0.5]=0
        # 正则化
        mask_gt = torchvision.transforms.ToTensor()(mask1)
        mask_512=mask_gt.clone()
        mask_512[mask_gt>=0.4]=1
        mask_512[mask_gt<0.4]=0
        
        edge = max_mask256-mask_512
        
        mask_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(mask_512).squeeze(0)
        mask_224[mask_224>=0.4]=1
        mask_224[mask_224<0.4]=0
        
        max_mask_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(max_mask256).squeeze(0)
        max_mask_224[max_mask_224>=0.5]=1
        max_mask_224[max_mask_224<0.5]=0
        
        edge_224 = max_mask_224-mask_224
        
        pose_224=torch.zeros(3,224,224)
        pose_224[0,:,:][edge_224==1]=20/255
        pose_224[1,:,:][edge_224==1]=80/255
        pose_224[2,:,:][edge_224==1]=194/255
        
        pose_224[0,:,:][mask_224==1]=248/255
        pose_224[1,:,:][mask_224==1]=251/255
        pose_224[2,:,:][mask_224==1]=14/255
        
        a = np.random.uniform()
        if self.augment_fg:
            aug2 = Transforms_PIL(Basic(), (256,256))
            fg_img, pose2, mask2, max_mask2 = aug2(img, pose, mask, max_mask)
        else:
            fg_img, pose2, mask2, max_mask2 = img, pose, mask, max_mask
        fg_img = torchvision.transforms.ToTensor()(fg_img)
        mask2 = torchvision.transforms.ToTensor()(mask2)
        mask2[mask2>=0.4]=1
        mask2[mask2<0.4]=0
 
        ref_mask_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(mask2).squeeze(0)
        ref_mask_224[ref_mask_224>=0.4]=1
        ref_mask_224[ref_mask_224<0.4]=0
        
        max_mask2 = torchvision.transforms.ToTensor()(max_mask2)
        ref_max_mask_224 = torchvision.transforms.Resize((224,224),Image.BILINEAR)(max_mask2).squeeze(0)
        ref_max_mask_224[ref_max_mask_224>=0.5]=1
        ref_max_mask_224[ref_max_mask_224<0.5]=0
        
        ref_edge_224 = ref_max_mask_224-ref_mask_224
        
        ref_pose_224=torch.zeros(3,224,224)
        ref_pose_224[0,:,:][ref_edge_224==1]=20/255
        ref_pose_224[1,:,:][ref_edge_224==1]=80/255
        ref_pose_224[2,:,:][ref_edge_224==1]=194/255
        
        ref_pose_224[0,:,:][ref_mask_224==1]=248/255
        ref_pose_224[1,:,:][ref_mask_224==1]=251/255
        ref_pose_224[2,:,:][ref_mask_224==1]=14/255
        
        if a > 0.05:
            ref = fg_img*max_mask2+(1-max_mask2) 
            refernce = torchvision.transforms.Resize((224,224),Image.BILINEAR)(ref)
            refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))(refernce)
            
        else:
            ref = torch.ones(3,256,256)
            ref_pose_224 = torch.zeros(3,224,224)
            refernce = torch.zeros(3,224,224)
        
        img_512 = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_512)

        # 生成 inpaint 和 hint
        
        inpaint = img_512 * (1-max_mask256)  #+ edge * torch.randn_like(img_512)
        hint1 = torch.cat((mask_512,edge), dim = 0)
        #hint1 = pose_256 #torch.cat((edge,mask_512,mask_512), dim = 0)
        
        return {"GT": img_512,                  
                "inpaint_image": inpaint,  
                "inpaint_mask":  1-max_mask256,      
                "ref_imgs": refernce,     
                "hint": hint1,        
                "real_mask": max_mask256,
                "GT_mask": mask_512,
                "pose": pose_224,
                "ref_pose": ref_pose_224
                }