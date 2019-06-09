#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from keras.models import load_model
import json
from glob import glob


get_ipython().run_line_magic('matplotlib', 'inline')


img_path = './data/crop_path_083_3/'
label_path = './data/cate_attr_label.csv'


def pad_image(image, target_size):
    iw, ih = image.size 
    w, h = target_size 
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image



# PARAMS
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES_ONE = 16
NUM_CLASSES_ATTR = 92


img_list = glob(img_path + "*.jpg")


model = load_model('./attr-epoch5.h5')



result = {}
cnt = 0
for path in img_list:
    img = Image.open(path)
    img = pad_image(img, (IMG_WIDTH, IMG_HEIGHT))
    arr = np.zeros((1,224,224,3))
    arr[0] = np.array(img)
    res = model.predict(arr,batch_size=1)
#     print(res)
    attr_indexes = []
    cate_index = []
    for i in range(92):
        if res[i][0][0]>0.5:
            attr_indexes.append(i)
    cate_index.append(np.argmax(res[92]))
#     print(attr_indexes, cate_index)
    result["%s" % path[23:]] = (attr_indexes, cate_index)
    cnt +=1
    if cnt %1000 ==0:
        print(cnt)
#     break



with  open('attr.json', 'w') as f:
    # to save int64, trans to string
    json.dump(str(result), f)




# with open('attr.json', 'r') as f: 
    # data= eval(json.load(f))
# data




