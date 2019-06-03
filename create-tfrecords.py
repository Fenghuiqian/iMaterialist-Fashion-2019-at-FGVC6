#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import cv2
import sys
import os
import hashlib
import io
import json
import contextlib2
import PIL.Image
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio
import json


tfrecords_path = 'data/tfrecords/'
train_path = "data/train/"
test_path = "data/test/"
label_describe_path = "data/label_descriptions.json"
annotation_path = "data/train.csv"


# train samples , test samples
print(len(os.listdir(train_path)), len(os.listdir(test_path)))


# requirements of the competition
def resize_bi_mask(binary_mask):
    return cv2.resize(binary_mask, (512, 512), cv2.INTER_NEAREST)


# run-length-encode
def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


# for submittion format
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# decode rle
def rle_decode(rle_str, mask_shape, mask_dtype):
    rle = [int(i) for i in rle_str.split()]
    mask = np.zeros(mask_shape[0]*mask_shape[1], dtype=mask_dtype)
    for starts,ends in zip(rle[::2], rle[1::2]):
        mask[starts-1:starts-1+ends] = 1
    return mask.reshape(mask_shape[::-1], order='F')
	
	
# extract bbox from mask
def extract_bboxes(mask):
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # box is at outside of object
        x2 += 1
        y2 += 1
    else:
        x1, x2, y1, y2 = 0, 0, 0, 0
    width = x2-x1
    height = y2-y1
    return (x1, y1, width, height)
	
# tfrecords required
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_examples_list(path):
    """Read list of training or validation examples.
    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.
    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).
    Args:
    path: absolute path to examples list file.
    Returns:
    list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
        return [line.strip().split(' ')[0] for line in lines]

def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=True):
    """Converts image and annotations to a tf.Example proto.

    Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0

    for object_annotations in annotations_list:
        if include_masks:
            run_len_encoding = object_annotations['segmentation']
            binary_mask = rle_decode(run_len_encoding, mask_shape=(image_width, image_height), mask_dtype =np.uint8 )
            if not object_annotations['iscrowd']:
                pil_image = PIL.Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                encoded_mask_png.append(output_io.getvalue())
        
        x, y, width, height = extract_bboxes(binary_mask)
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(width*height)


    
    feature_dict = {
          'image/height':
              int64_feature(image_height),
          'image/width':
              int64_feature(image_width),
          'image/filename':
              bytes_feature(filename.encode('utf8')),
          'image/source_id':
              bytes_feature(str(image_id).encode('utf8')),
          'image/key/sha256':
              bytes_feature(key.encode('utf8')),
          'image/encoded':
              bytes_feature(encoded_jpg),
          'image/format':
              bytes_feature('jpeg'.encode('utf8')),
          'image/object/bbox/xmin':
              float_list_feature(xmin),
          'image/object/bbox/xmax':
              float_list_feature(xmax),
          'image/object/bbox/ymin':
              float_list_feature(ymin),
          'image/object/bbox/ymax':
              float_list_feature(ymax),
          'image/object/class/text':
              bytes_list_feature(category_names),
          'image/object/is_crowd':
              int64_list_feature(is_crowd),
          'image/object/area':
              float_list_feature(area)
                   }
    if include_masks:
        feature_dict['image/object/mask'] = (bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped
	


# load categories file
with open(label_describe_path) as json_data:
    data = json.load(json_data)
category_index = data['categories']

# load annotation file
df_annotation = pd.read_csv(annotation_path)
# df_annotation.head()
attributed_img_index = [] 
for i in range(df_annotation.shape[0]):
    if '_' in df_annotation.loc[i, 'ClassId']:
        attributed_img_index.append(i)
# drop attributed rows
df_annotation.drop(attributed_img_index, axis=0, inplace=True)
print(df_annotation['ImageId'].unique().shape)



# def tfrecords writer
train_writer = tf.python_io.TFRecordWriter(tfrecords_path + 'train_records')
val_writer = tf.python_io.TFRecordWriter(tfrecords_path + 'val_records')


t1 = time.time()
image_id = 0
for each_image_id in df_annotation['ImageId'].unique():
    annotations_list = []
    image_dict = {}
    temp = df_annotation[df_annotation['ImageId']==each_image_id].reset_index()
    # image info
    image_dict['file_name'] = temp.loc[0,'ImageId']
    image_dict['id'] = image_id
    image_dict['height'] = temp.loc[0,'Height']
    image_dict['width'] = temp.loc[0,'Width']
    # annotations
    for i in range(temp.shape[0]):
        each_annote = {}
        each_annote['segmentation'] = temp.loc[i, 'EncodedPixels']
        each_annote['image_id'] = image_id
        each_annote['iscrowd'] = 0
        each_annote['category_id'] = temp.loc[i, 'ClassId']
        annotations_list.append(each_annote)
    # image dir
    image_dir = train_path
    # tf example
    key, tf_example, num_annotations_skipped = create_tf_example(image_dict,
														  annotations_list,
														  image_dir,
														  category_index,
														  include_masks=True)
    
    if image_id < 40000:
        # write train tf records ,train nums=40000
        train_writer.write(tf_example.SerializeToString())
    else:
        # write val tf records, val nums=5620
        val_writer.write(tf_example.SerializeToString())
    if image_id % 2000 ==0:
        print(image_id)
    image_id +=1

t2 = time.time()
print('cost',(t2-t1)/3600, 'h')






