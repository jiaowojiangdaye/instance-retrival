
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
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
#config.gpu_options.per_process_gpu_memory_fraction = 1 
set_session(tf.Session(config=config))

#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("")
CUR_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import clothes

get_ipython().run_line_magic('matplotlib', 'inline')

# Path to trained weights file

# Directory to save logs and trained model
LOGS_DIR = os.path.join(CUR_DIR, "logs")

computer = 'jz'  #zy jz 426

if computer == '426':
    COCO_WEIGHTS_PATH = '/raid/Guests/DaYea/clothes/mask_rcnn_coco.h5'
    IMG_DIR = '/raid/Guests/Jay/Jay/datasets/clothes/Img'
    annotations_path = os.path.join("/raid/Guests/zy/clothes/Anno/cloth_all.csv")
    class_path =  os.path.join("/raid/Guests/zy/clothes/Anno/list_class.csv")
    img_path = '/raid/Guests/zy/clothes/Anno/cloth.h5'
    mask_path = r'/raid/Guests/zy/clothes/Anno/mask.h5'
    
elif computer == 'jz':
    COCO_WEIGHTS_PATH = '/hdisk/Ubuntu/backup/aWS/obj-detection/Mask_RCNN-master/model/mask_rcnn_coco.h5'
    IMG_DIR = r"/hdisk/Ubuntu/datasets/clothes/Img"
    annotations_path = r"/hdisk/Ubuntu/datasets/clothes/Anno/cloth_all.csv"
    class_path =  r"/hdisk/Ubuntu/datasets/clothes/Anno/list_class.csv"
    img_path = r"/hdisk/Ubuntu/datasets/clothes/Anno/cloth.h5"
    mask_path = r"/hdisk/Ubuntu/datasets/clothes/Anno/mask.h5"
    
elif computer == 'zy':
    
    data_root = '/media/mosay/数据/jz/dataset'
#    IMG_DIR = os.path.join(ROOT_DIR, "clothes/Img/")
    COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
    IMG_DIR =  data_root + '/Img'
    annotations_path = data_root + "/Anno/cloth_all.csv"
    class_path =   data_root + "/Anno/list_class.csv"
    img_path =  data_root + "/Anno/cloth.h5"
    mask_path =  data_root + "/Anno/mask.h5"
    
    
# In[2]:

#tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 3 # 分配50%  
#session = tf.Session(config=tf_config)  



# In[3]:


config = clothes.ClothConfig()
config.display()


# In[4]:


# Training dataset.
dataset_train = clothes.ClothDataset()
dataset_train.load_cloth(config, annotations_path, class_path, img_path, mask_path, IMG_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = clothes.ClothDataset()
dataset_val.load_cloth(config, annotations_path, class_path, img_path, mask_path, IMG_DIR, "val")
dataset_val.prepare()


# In[5]:


# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])


# In[6]:

model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIR)
# In[7]:

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "this":
    model.load_weights('/home/aaron/mydisk/aWS/ImgRetrival/logs/mask_rcnn_clothes_0464.h5', by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
#    model.load_weights(LOGS_DIR+'/mask_rcnn_clothes.h5', by_name=True)
    model.load_weights(model.find_last(), by_name=True)



# In[8]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
    
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS,
            augmentation=None,
            layers='heads')


# In[8]:


model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS,
            augmentation=None,
            layers='all')


# In[9]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(LOGS_DIR, "mask_rcnn_clothes.h5")
model.keras_model.save_weights(model_path)

