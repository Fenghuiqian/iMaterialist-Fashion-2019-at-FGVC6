#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import PIL
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import pandas as pd
sys.path.append("./object/")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from glob import glob

get_ipython().run_line_magic('matplotlib', 'inline')


PATH_TO_FROZEN_GRAPH = "./models/frozen_inference_graph.pb"
PATH_TO_LABELS = os.path.join('./', 'imaterialist_fashion_label_map.pbtxt')
CROP_IMG_PATH = "./cropped_img/"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

          # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def resize_bi_mask(binary_mask, target_shape):
    return cv2.resize(binary_mask, target_shape, cv2.INTER_NEAREST)


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


# Detection

img_list = glob("./data/test/*.jpg")

# save result
result = []
for image_path in img_list:
    image = Image.open(image_path)
    image_width = image.size[0]
    image_height = image.size[1]
    target_width = image_width
    target_height = image_height
    if image_width >= image_height and image_width > 1280: # > 1365/800:
        target_width = 1280
        target_height = int(1280*image_height/image_width)
        image = image.resize((target_width, target_height), PIL.Image.ANTIALIAS)
    elif image_height > image_width and image_height > 1280:
        target_height = 1280
        target_width = int(1280*image_width/image_height)
        image = image.resize((target_width, target_height), PIL.Image.ANTIALIAS)

  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
  # detect
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    
    img_dict = {}
    for num in range(output_dict["num_detections"]):
        mask = output_dict["detection_masks"][num]
        mask = resize_bi_mask(mask, (512,512))
        rle = rle_encode(mask)
        img_dict["EncodedPixels"] = " ".join(list(map(str, rle)))
        img_dict["ImageId"] = image_path[12:]
        img_dict["ClassId"] = output_dict["detection_classes"][num]
        result.append(img_dict)
		
		# save bbox cropped  images
        bbox_img_coor = output_dict["detection_boxes"][num]
        y_min, x_min, y_max, x_max = bbox_img_coor[0] * target_height, bbox_img_coor[1] * target_width, bbox_img_coor[2] * target_height, bbox_img_coor[3] * target_width
        bbox_croped_img = image.crop(box=[x_min, y_min, x_max, y_max])
        bbox_croped_img.save(CROP_IMG_PATH +"%s" % num + image_path[34:])
		
    # handle no object imgs, the competetion required
    if output_dict["num_detections"] == 0:
        img_dict["EncodedPixels"] = "1 1"
        img_dict["ImageId"] = image_path[12:]
        img_dict["ClassId"] = 22


submit = pd.DataFrame.from_records(result, columns=["ImageId", "EncodedPixels", "ClassId"])
submit.to_csv("submit.csv")
