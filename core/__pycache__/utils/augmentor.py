import numpy as np
import random
import warnings
import os
import time
from glob import glob
from skimage import color, io
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F

def get_middlebury_images():
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), 'r') as f:
        lines = f.read().splitlines()
    return sorted([os.path.join(root, 'trainingQ', f'{name}/im0.png') for name in lines])

def get_eth3d_images():
    return sorted(glob('datasets/ETH3D/two_view_training/*/im0.png'))

def get_kitti_images():
    return sorted(glob('datasets/KITTI/training/image_2/*_10.png'))

def transfer_color(image, style_mean, style_stddev):
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0,1), keepdims=True)# + 1
    reference_mean = np.mean(reference_image_lab, axis=(0,1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev/reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l,a,b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb

class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"



def get_cropped_H_inv1(H_inv: np.ndarray, x0: float, y0: float) -> np.ndarray:
    T_crop = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ], dtype=np.float64)

    T_crop_inv = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ], dtype=np.float64)

    return T_crop @ H_inv @ T_crop_inv

def get_scaled_H_inv1(H_inv, scale_x, scale_y):
    H_inv_64 = H_inv.astype(np.float64)

    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    S_inv = np.array([
        [1.0 / scale_x, 0, 0],
        [0, 1.0 / scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # 正确的变换顺序
    H_inv_scaled = S @ H_inv_64 @ S_inv

    # 不要修改最后一行！保持原始的透视结构
    # 返回时保持精度
    return H_inv_scaled  # 先不转换回float32，看看效果

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow_h,flow_v,flow,H_inv):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow_h = cv2.resize(flow_h, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow_v = cv2.resize(flow_v, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = flow * [scale_x, scale_y]
            flow_h = flow_h * scale_x
            flow_v =flow_v*scale_y
            H_inv = get_scaled_H_inv1(H_inv, scale_x, scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow_h = flow_h[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow_v = flow_v[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            H_inv=get_cropped_H_inv1(H_inv,x0, y0)


        return img1, img2, flow_h,flow_v,flow,H_inv


    def __call__(self, img1, img2, flow_h,flow_vertical,flow,H_inv):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow_h,flow_v,flow_rect,H_inv_new = self.spatial_transform(img1, img2, flow_h,flow_vertical,flow,H_inv)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow_h = np.ascontiguousarray(flow_h)
        flow_v = np.ascontiguousarray(flow_v)
        flow_rect = np.ascontiguousarray(flow_rect)
        H_inv_new = np.ascontiguousarray(H_inv_new)

        return img1, img2, flow_h,flow_v,flow_rect,H_inv_new

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7,1.3], gamma=[1,1,1,1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def get_scaled_H_inv(self,H_inv, scale_x, scale_y):
        """根据缩放参数调整H_inv矩阵"""
        # S = np.array([
        #     [scale_x, 0, 0],
        #     [0, scale_y, 0],
        #     [0, 0, 1]
        # ], dtype=np.float64)
        #
        # S_inv = np.array([
        #     [1 / scale_x, 0, 0],
        #     [0, 1 / scale_y, 0],
        #     [0, 0, 1]
        # ], dtype=np.float64)
        #
        # return S @ H_inv @ S_inv
        H_inv_64 = H_inv.astype(np.float64)

        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        S_inv = np.array([
            [1.0 / scale_x, 0, 0],
            [0, 1.0 / scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        # 正确的变换顺序
        H_inv_scaled = S @ H_inv_64 @ S_inv

        # 不要修改最后一行！保持原始的透视结构
        # 返回时保持精度
        return H_inv_scaled  # 先不转换回float32，看看效果

    def get_cropped_H_inv(self,H_inv, x0, y0):
        """根据裁剪参数调整H_inv矩阵"""
        T_crop = np.array([
            [1, 0, -x0],
            [0, 1, -y0],
            [0, 0, 1]
        ], dtype=np.float64)

        T_crop_inv = np.array([
            [1, 0, x0],
            [0, 1, y0],
            [0, 0, 1]
        ], dtype=np.float64)

        return T_crop @ H_inv @ T_crop_inv

    def resize_sparse_flow_map(self, flow_h=None, valid_h=None, fx=1.0, fy=1.0, flow_vertical=None, valid_vertical=None, flow=None, valid=None):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        flow_h = flow_h.reshape(-1, 2).astype(np.float32)
        valid_h = valid_h.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]   #这里valid是有效值，不是视差值
        flow0 = flow[valid>=1]

        coords0_h = coords[valid_h >= 1]  # 这里valid是有效值，不是视差值
        flow0_h = flow_h[valid_h >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        coords1_h = coords0_h * [fx, fy]
        flow1_h = flow0_h * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx >= 0) & (xx < wd1) & (yy >=0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        #这个地方开始水平方向视差
        xx_h = np.round(coords1_h[:, 0]).astype(np.int32)
        yy_h = np.round(coords1_h[:, 1]).astype(np.int32)

        v_h = (xx_h >= 0) & (xx_h < wd1) & (yy_h >= 0) & (yy_h < ht1)
        xx_h = xx_h[v_h]
        yy_h = yy_h[v_h]
        flow1_h = flow1_h[v_h]


        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img_h = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img_h = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        flow_img_h[yy_h, xx_h] = flow1_h
        valid_img_h[yy_h, xx_h] = 1

        if flow_vertical is not None and valid_vertical is not None:
            flow_vertical = flow_vertical.reshape(-1, 2).astype(np.float32)
            valid_vertical = valid_vertical.reshape(-1).astype(np.float32)
            coords0_vertical = coords[valid_vertical >= 1]
            flow0_vertical = flow_vertical[valid_vertical >= 1]

            coords1_vertical = coords0_vertical * [fx, fy]
            flow1_vertical = flow0_vertical * [fx, fy]

            xx_vertical = np.round(coords1_vertical[:, 0]).astype(np.int32)
            yy_vertical = np.round(coords1_vertical[:, 1]).astype(np.int32)
            v_vertical = (xx_vertical >= 0) & (xx_vertical < wd1) & (yy_vertical >= 0) & (yy_vertical < ht1)
            xx_vertical = xx_vertical[v_vertical]
            yy_vertical = yy_vertical[v_vertical]
            flow1_vertical = flow1_vertical[v_vertical]

            flow_vertical_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
            valid_vertical_img = np.zeros([ht1, wd1], dtype=np.int32)
            flow_vertical_img[yy_vertical, xx_vertical] = flow1_vertical
            valid_vertical_img[yy_vertical, xx_vertical] = 1

            return flow_img_h, valid_img_h, flow_vertical_img, valid_vertical_img,flow_img, valid_img
        else:
            return flow_img, valid_img

        #return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow_h=None, valid_h=None,flow_vertical=None,valid_vertical=None,flow=None,valid=None,H_inv=None):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            H_inv = self.get_scaled_H_inv(H_inv, scale_x, scale_y)
            flow_h, valid_h,flow_vertical,valid_vertical,flow,valid = self.resize_sparse_flow_map(flow_h=flow_h, valid_h=valid_h, fx=scale_x, fy=scale_y, flow_vertical=flow_vertical, valid_vertical=valid_vertical,flow=flow, valid=valid)



        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip   #这个地方没有仔细推敲
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                flow_vertical = flow_vertical[:, ::-1] * [-1.0, 1.0]
                flow_h = flow_h[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                flow_vertical = flow_vertical[::-1, :] * [1.0, -1.0]
                flow_h = flow_h[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        H_inv = self.get_cropped_H_inv(H_inv, x0, y0)

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow_h = flow_h[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow_vertical = flow_vertical[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_h = valid_h[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_vertical = valid_vertical[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        
        return img1, img2, flow_h, valid_h,flow_vertical, valid_vertical,flow, valid,H_inv


    def __call__(self, img1, img2, flow_h, valid_h,flow_vertical,valid_vertical,flow,valid,H_inv):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow_h, valid_h,flow_vertical,valid_vertical,flow,valid,H_inv= self.spatial_transform(img1, img2, flow_h=flow_h, valid_h=valid_h,flow_vertical=flow_vertical,valid_vertical=valid_vertical,flow=flow,valid=valid,H_inv=H_inv)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow_h = np.ascontiguousarray(flow_h)
        flow_vertical = np.ascontiguousarray(flow_vertical)
        flow = np.ascontiguousarray(flow)
        valid_h = np.ascontiguousarray(valid_h)
        valid_vertical = np.ascontiguousarray(valid_vertical)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow_h, valid_h, flow_vertical, valid_vertical,flow, valid,H_inv
