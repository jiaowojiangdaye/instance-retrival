
# coding: utf-8

# In[1]:

import os 
import tensorflow as tf 
import keras.backend.tensorflow_backend as ktf 
# GPU 显存自动分配 
config = tf.ConfigProto() 
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3 
session = tf.Session(config=config) 
ktf.set_session(session) 
# 指定GPUID, 第一块GPU可用 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
os.system('echo $CUDA_VISIBLE_DEVICE')
import sys
#import random
#import math
#import re
#import time
#import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session   
config.gpu_options.allow_growth = True  
#config.gpu_options.per_process_gpu_memory_fraction = 1 
set_session(tf.Session(config=config))

#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))


#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("")
CUR_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn import utils
#from mrcnn import visualize
#from mrcnn.visualize import display_images
import mrcnn.model as modellib
#from mrcnn.model import log
import ThisConfig

#get_ipython().run_line_magic('matplotlib', 'inline')

#tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 3 # 分配50%  
#session = tf.Session(config=tf_config)


config = ThisConfig.ThisConfig()
config.display()


# In[4]:


# Training dataset.
dataset_train = ThisConfig.ThisDataset()
dataset_train.load_data(config, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = ThisConfig.ThisDataset()
dataset_val.load_data(config, "val")
dataset_val.prepare()

# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
affine_range = 35
augmentation = iaa.SomeOf((0, 4), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
#    iaa.Crop(px=(0, 50)), #woshi xiao cao
    iaa.OneOf([iaa.Affine(rotate=(0-affine_range, 0+affine_range)),
               iaa.Affine(rotate=(90-affine_range, 90+affine_range)),
               iaa.Affine(rotate=(180-affine_range, 180+affine_range)),
               iaa.Affine(rotate=(270-affine_range, 270+affine_range))]),
#    iaa.Multiply((0.8, 1.5)), 
#    iaa.GaussianBlur(sigma=(0.0, 1.0))
    iaa.CropAndPad(px=(-100,100),
               percent=None,
               pad_mode='constant',
               pad_cval=0,
               keep_size=True,
               sample_independently=True,
               name=None,
               deterministic=False,
               random_state=None),
    iaa.Affine(scale=(0.40, 2)),
    iaa.Affine(translate_percent=None,
           translate_px={"x": (-100, 100), "y": (-100, 100)},
           rotate=0,
           shear=(-40, 40),
           order=[0, 1, 3, 4, 5],
           cval=0,
           mode='constant',
           name=None, deterministic=False, random_state=None),
#    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
#                         children=iaa.SomeOf((0,3),
#                                             [iaa.WithChannels(0, iaa.Add((-30, 30))),
#                                              iaa.WithChannels(1, iaa.Add((-30, 30))),
#                                              iaa.WithChannels(2, iaa.Add((-30, 30)))])),
    iaa.SomeOf((0,3),[iaa.WithChannels(0, iaa.Add((-40, 40))),  
                      iaa.WithChannels(1, iaa.Add((-40, 40))),
                      iaa.WithChannels(2, iaa.Add((-40, 40)))]),
])

model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.DEFAULT_LOGS_DIR)


if config.init_with == "this":
    model.load_weights(config.THIS_WEIGHT_PATH, by_name=True)
elif config.init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(config.COCO_WEIGHTS_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask", "rpn_model"])
elif config.init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

# Training - Stage 1
#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE,
#            epochs=config.EPOCHS,
#            augmentation=augmentation,
#            layers='rpn')
    
# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
#print("Fine tune Resnet stage 4 and up")
#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE,
#            epochs=config.EPOCHS,
#            layers='3+',
#            augmentation=augmentation)

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS,
            layers='all',
            augmentation=augmentation)



# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
#model_path = os.path.join(model_dir=config.DEFAULT_LOGS_DIR, "mask_rcnn_clothes.h5")
#model.keras_model.save_weights(model_path)

