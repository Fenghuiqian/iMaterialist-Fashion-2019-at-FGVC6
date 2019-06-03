#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob
from keras.applications.mobilenet_v2 import MobileNetV2
import os
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import Sequence
from PIL import Image
import os
import random
from imgaug import augmenters as iaa
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
import scipy.misc as misc
import imgaug as ia
import cv2
import time
np.random.seed(1024)
%matplotlib inline


img_path = './data/bbox-croped-image/'
label_path = './data/cate_attr_label.csv'


# pad image to fixed size
def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
 
    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)
 
    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
 
    return new_image


# generator of batch-multitask 
def generator_batch_multitask(data, num_class_one=16, num_class_attr=92, batch_size=16, return_label=True,
                            img_width=299, img_height=299, shuffle=False,
                            save_to_dir=None, augment=True):

    base_img_path ='./data/bbox-croped-image/'
    N = data.shape[0]
    
    # label encoding
    label_one_encoder = LabelEncoder().fit(data.cate)
    cls_one = label_one_encoder.transform(data.cate)
    label_encoder_attrs = LabelEncoder().fit(["YES","NO"])
    
    if shuffle:
        random.shuffle(data)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_one = np.zeros((current_batch_size, num_class_one))
        
        Y_batch_attr = np.zeros((num_class_attr, current_batch_size, 2))
        
        
        
        
        for i in range(current_index, current_index + current_batch_size):
#             line = data_list[i].strip().split(' ')
            #print line
            img_path = base_img_path + "attr_images" + data.loc[i, "image_name"]
            img = Image.open(img_path)
            img = pad_image(img, (img_width, img_height))

            X_batch[i - current_index] = img
            if return_label:
                label_one = cls_one[i]
                Y_batch_one[i - current_index, label_one] = 1
                
                # label of attr
                attr_list = eval(data.loc[i, "attr"])
                for attribute in range(num_class_attr):
                    if attribute in attr_list:
                        Y_batch_attr[attribute][i - current_index][0] = label_encoder_attrs.transform(["YES", "NO"])[0]
                    else:
                        Y_batch_attr[attribute][i - current_index][1] = label_encoder_attrs.transform(["NO", "YES"])[1]


        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)
        
        # return X
        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)
        
        # return Y
        attr_res = []
        for i in range(Y_batch_attr.shape[0]):
            attr_res.append(Y_batch_attr[i])
        attr_res.append(Y_batch_one)
        
        if return_label:
            yield ([X_batch], attr_res)
        else:
            yield X_batch
			


# PARAMS
IMG_WIDTH = 224
IMG_HEIGHT = 224
NBR_EPOCHS = 20
BATCH_SEZE= 4
NBR_TRAIN_SAMPLES = 10000
NBR_VALIDATION_SAMPLES = 1540
NUM_CLASSES_ONE = 16
NUM_CLASSES_ATTR = 92


# read samples & shuffle samples & split train/val 
data = pd.read_csv(label_path)
train_data = data[:NBR_TRAIN_SAMPLES]
val_data = data[NBR_TRAIN_SAMPLES:]


# load mobilenet_v2 model & modify outputs
def mobilenet_v2_model():
    print("downloading model...")
    MobileNetV2_notop = MobileNetV2(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape=(224, 224, 3), pooling='avg')
    print("done.")
    output = MobileNetV2_notop.get_layer(index = -1).output  # shape(None,2048)
    f_dense = Dense(512, name='f_acs')(output)
	
    f_category = Dense(NUM_CLASSES_ONE, activation='softmax', name='categories')(f_dense)

    # output heads of 92 attrs
    outputs = []
    for i in range(NUM_CLASSES_ATTR):
        outputs.append(Dense(2, activation='softmax', name='attr_%s' % i)(f_dense))
	
	# output heads of 1 category
    outputs.append(f_category)
	
	
    model = Model(inputs=MobileNetV2_notop.input, outputs=outputs)
    
    # loss funcs
    loss_func = []
    for i in range(NUM_CLASSES_ATTR):
        loss_func.append("binary_crossentropy")
    loss_func.append("categorical_crossentropy")
    
    # metrics
    metrics = ['accuracy']*(NUM_CLASSES_ATTR + 1)
    
    model.compile(loss=loss_func, optimizer = "adam", metrics = metrics)
    return model


# generators
train_generator = generator_batch_multitask(train_data, num_class_one=NUM_CLASSES_ONE, num_class_attr=NUM_CLASSES_ATTR, batch_size=BATCH_SEZE, return_label=True,
                            img_width=IMG_WIDTH, img_height = IMG_HEIGHT, shuffle=False,
                            save_to_dir=None, augment=False)

validation_generator = generator_batch_multitask(val_data, num_class_one=NUM_CLASSES_ONE, num_class_attr=NUM_CLASSES_ATTR, batch_size=BATCH_SEZE, return_label=True,
                            img_width=IMG_WIDTH, img_height = IMG_HEIGHT, shuffle=False,
                            save_to_dir=None, augment=False)
							



# fit 
model = mobilenet_v2_model()
model_file_saved = "./models/MobilenetV2_fine_tuned"
checkpoint = ModelCheckpoint(model_file_saved, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
tensorboard = TensorBoard(log_dir='./models/log', write_graph=True, write_grads=True, write_images=True, 
                                    embeddings_freq=1, update_freq=32)
model.fit_generator(
        train_generator,
        samples_per_epoch = NBR_TRAIN_SAMPLES,
        nb_epoch = NBR_EPOCHS,
        validation_data = validation_generator,
        nb_val_samples = NBR_VALIDATION_SAMPLES,
        callbacks = [checkpoint, early_stop, tensorboard],
        verbose=1)


model.save("mobilev2-attr.h5")










