# iMaterialist-Fashion-2019-at-FGVC6
Fine-grained segmentation task for fashion and apparel
# Usage
1. Install TensorFlow Object Detection API.  More at https://github.com/tensorflow/models
2. Run create-tfrecords.py to get tfrecords file.
3. Train the Object Detection model.
4. Run mask-detect.py to get predicted mask and save object bbox croped images.
5. Run attr-manage/create-attribute-train-val-file.py to create attribute tarin images.
6. Run attr-manage/attr-categories-train.py to train the attribute classification model.
7. Run attr-manage/attr-predict.py to predict attribute result for the object bbox croped images in step 4.
# Notes
LeaderBoard Top17%(43/242).
