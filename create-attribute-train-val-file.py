#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import cv2
import os
# print(os.listdir("../input"))
import hashlib
import io
import json
import contextlib2
import PIL.Image
from PIL import ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio
import json
import random


root_path = 'attr_images'
train_path = "data/train/"
test_path = "data/test/"
annotation_path = "data/train.csv"


# train samples , test samples
len(os.listdir(train_path)), len(os.listdir(test_path))



# extract bbox from mask 
def extract_bboxes(mask):
    #Bounding box. 
    #返回mask元素值为1的横纵坐标
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        #x2和y2不应该作为box的一部分. 所以各自加1.
        x2 += 1
        y2 += 1
    else:
        #该instance没有mask. 可能是由于缩放和裁剪导致的
        #将bbox设为0
        x1, x2, y1, y2 = 0, 0, 0, 0
    width = x2-x1
    height = y2-y1
    return (x1, y1, width, height)
	
	

# decode rle
def rle_decode(rle_str, mask_shape, mask_dtype):
    rle = [int(i) for i in rle_str.split()]
    mask = np.zeros(mask_shape[0]*mask_shape[1], dtype=mask_dtype)
    for starts,ends in zip(rle[::2], rle[1::2]):
        mask[starts-1:starts-1+ends] = 1
    return mask.reshape(mask_shape[::-1], order='F')
	
	

# create attr-annotation info
df_annotation = pd.read_csv(annotation_path)
attributed_img_index = [] 
for i in range(df_annotation.shape[0]):
    if '_' in df_annotation.loc[i, 'ClassId']:
        attributed_img_index.append(i)

attr_annotation = df_annotation.iloc[attributed_img_index, :]
attr_annotation.shape



# crop images by bbox then save it. 
cnt=0
cate_attr_label = []
for ind in attr_annotation.index:
    # get bbox by rle mask
    rle = attr_annotation.loc[ind, 'EncodedPixels']
    img_width = attr_annotation.loc[ind, 'Width']
    img_height = attr_annotation.loc[ind, 'Height']
    binary_mask = rle_decode(rle, mask_shape=(img_width, img_height), mask_dtype =np.uint8 )
    x, y, width, height = extract_bboxes(binary_mask)
    
    # crop bbox from image
    image_id = attr_annotation.loc[ind, 'ImageId']
    img = PIL.Image.open(train_path + image_id)
    bbox_croped_img = img.crop(box=[x,y,x+width,y+height])
    # create image name
    image_name = str(random.randint(1,10000000)) + image_id
    bbox_croped_img.save(root_path + image_name)
    # attributions
    class_id = attr_annotation.loc[ind, 'ClassId'].split('_')
    cate, attr = int(class_id[0]), class_id[1:]
    attr = sorted([int(i) for i in attr])
    record = {'image_name':image_name, 'cate':cate, 'attr':attr}
    cate_attr_label.append(record)
    if cnt % 100 == 0:
        print(cnt)
    cnt += 1



# save cate-attr-file
file = pd.DataFrame.from_records(cate_attr_label, columns = ['image_name', 'cate', 'attr'])
file.to_csv('cate_attr_label.csv', index=None, header=True)




