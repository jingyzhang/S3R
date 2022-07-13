#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/9 3:42 PM
# @Author  : Jingyang.Zhang
'''
signed distance function
'''
import numpy as np
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
import cv2
import matplotlib.pyplot as plt

def compute_sdf(img_gt):

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros([img_gt.shape[0], 1, img_gt.shape[2], img_gt.shape[3]])

    for b in range(normalized_sdf.shape[0]): # batch size
        posmask = img_gt[b,1,:,:].astype(np.bool)
        negmask = img_gt[b,0,:,:].astype(np.bool)
        if posmask.any():
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b,0,:,:] = sdf

    return normalized_sdf