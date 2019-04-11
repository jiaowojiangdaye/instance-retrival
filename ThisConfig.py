#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:07:08 2019

@author: aaron
"""

import os
import sys
import json
#import datetime
import time
#import h5py
import numpy as np
import pandas as pd
#import skimage.io
#from imgaug import augmenters as iaa
from keras.preprocessing import image
#from PIL import Image
# Root directory of the project

from mrcnn.config import Config
from mrcnn import utils
#from mrcnn import model as modellib
#from mrcnn import visualize

from PIL import Image,ImageDraw,ImageFont
from tools import check_path, scan_specified_files
import random

############################################################
#  Configurations
############################################################
class ThisConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    
    #%%   about path
    
    annotion_path = '../../data/train/jinnan2_round1_train_20190305/train_no_poly.json'
    train_img_dir = '../../data/train/train'
    val_img_dir = '../../data/train/val'
    restricted_img_dir = '../../data/train/jinnan2_round1_train_20190222/restricted'
    normal_img_dir = '../../data/train/jinnan2_round1_train_20190222/normal'
    
    test_img_dir = '../../data/test/jinnan2_round1_test_a_20190306'
    real_test_img_dir = '../../data/test/jinnan2_round1_test_a_20190306'
#    real_test_img_dir = '../../data/test/error'
#    real_test_img_dir = '../../data/test/single'
    real_test_img_dir = '/media/mosay/数据/jz/tianchi/data/test/final/jinnan2_round1_test_b_20190326'
    
    
    
    computer = 'jz'  #zy jz 426

    if computer == '426':
        COCO_WEIGHTS_PATH = '/raid/Guests/DaYea/clothes/mask_rcnn_coco.h5'
        IMG_DIR = '/raid/Guests/Jay/Jay/datasets/clothes/Img'
        annotations_path = os.path.join("/raid/Guests/zy/clothes/Anno/cloth_all.csv")
        class_path =  os.path.join("/raid/Guests/zy/clothes/Anno/list_class.csv")
        img_path = '/raid/Guests/zy/clothes/Anno/cloth.h5'
        mask_path = r'/raid/Guests/zy/clothes/Anno/mask.h5'
        
    elif computer == 'jz':
        COCO_WEIGHTS_PATH = 'models/mask_rcnn_coco.h5'
        IMG_DIR = r"/hdisk/Ubuntu/datasets/clothes/Img"
        annotations_path = r"/hdisk/Ubuntu/datasets/clothes/Anno/cloth_all.csv"
        class_path =  r"/hdisk/Ubuntu/datasets/clothes/Anno/list_class.csv"
        img_path = r"/hdisk/Ubuntu/datasets/clothes/Anno/cloth.h5"
        mask_path = r"/hdisk/Ubuntu/datasets/clothes/Anno/mask.h5"
        query_file = '/home/aaron/mydisk/datasets/from_zy/MVC/query_image'
        
    elif computer == 'zy':
        
        data_root = '/media/mosay/数据/jz/dataset'
    #    IMG_DIR = os.path.join(ROOT_DIR, "clothes/Img/")
        COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
        IMG_DIR =  data_root + '/Img'
        annotations_path = data_root + "/Anno/cloth_all.csv"
        class_path =   data_root + "/Anno/list_class.csv"
        img_path =  data_root + "/Anno/cloth.h5"
        mask_path =  data_root + "/Anno/mask.h5"
        query_file = data_root + '/MVC/query_image'
    
    
    
    #%%  about dataset
    
    
    
    NAME = "clothes"
    
    
    # Number of classes (including background)
    NUM_CLASSES = 46 + 1  # Background + class
    split_record_dir = 'split_record'
    VAL_DATA_RATE = 0.1
    
    # Path to trained weights file
    DEFAULT_LOGS_DIR = 'logs'
    #%% important args
    Mode='retrival' #'train' or 'evaluate' or 'retrival'
#    Mode='evaluate'
    #%% evaluate
    if Mode == 'evaluate' or Mode == 'retrival':
        if computer =='jz':
            retrival_data_root = '/home/aaron/mydisk/datasets/from_zy'
        elif computer == 'zy':
            retrival_data_root = '/media/mosay/数据/jz/dataset'
        else:
            retrival_data_root = ''
        subset = 'mini_subset' #mini_subset  subset  
        if subset == 'subset':
            DEFAULT_DATASET = os.path.join(retrival_data_root, 'MVC/subset_annotation.txt')
        if subset == 'mini_subset':
            DEFAULT_DATASET = os.path.join(retrival_data_root, 'MVC/mini_subset_annotation.txt')#
        
        USING_NEGATIVE_IMG = True  # uesing img have no objects
        USING_POSITIVE_IMG = False  # uesing img have objects
        NEGATIVE_MULT = 1
        POSITIVE_MULT = 1
        real_test = True # if true means that we load image without gt
        init_with = "this"  # imagenet, coco, or last this
        EVA_LIMIT=100000
        
        model_version = 'mask_rcnn_clothes_0742'
        
        THIS_WEIGHT_PATH = 'models/mask_rcnn_clothes_0742.h5'  
        if init_with == 'coco':
            NUM_CLASSES = 80+1  # clothes has 80 classes
    
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.5
        # Skip detections with < 60% confidence
        DETECTION_MIN_CONFIDENCE =0.01
        # Non-maximum suppression threshold for detection
        DETECTION_NMS_DIFF_CLS = True   #!!!!!!!!! important
        DETECTION_NMS_THRESHOLD = 0.1   #!!!!!!!!! important
        POST_NMS_ROIS_INFERENCE = 1000   #!!!!!!!!! important
        map_iou_thr = 0.7
        arg_str = '_rn'+str(RPN_NMS_THRESHOLD)[2:4] +\
                  '_ds'+str(DETECTION_MIN_CONFIDENCE)[2:4] +\
				  '_dn'+str(DETECTION_NMS_THRESHOLD)[2:4]

        save_base_dir='test_' + model_version +'_'+ str(real_test)+'_' + arg_str
        check_path(save_base_dir)
        

        
    #%%train
    if Mode == 'train':
        USING_NEGATIVE_IMG = True
        USING_POSITIVE_IMG = True
        NEGATIVE_MULT = 1
        POSITIVE_MULT = 1
        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last this
        THIS_WEIGHT_PATH = '/home/aaron/mydisk/aWS/ImgRetrival/logs/mask_rcnn_clothes_0464.h5'
        COCO_WEIGHTS_PATH = 'models/mask_rcnn_coco.h5'
        # Learning rate and momentum
        # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
        # weights to explode. Likely due to differences in optimizer
        # implementation.
        LEARNING_RATE = 0.0001
        LEARNING_MOMENTUM = 0.9
        # Weight decay regularization
        WEIGHT_DECAY = 0.0001
        
        # Uncomment to train on 8 GPUs (default is 1)
        GPU_COUNT = 1
        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1
        # Number of training steps per epoch
        STEPS_PER_EPOCH = 300
        VALIDATION_STEPS = 50
        EPOCHS = 2000
        
        
        USE_RPN_ROIS = True
        rpn_fg_iou_thr = 0.5
        rpn_bg_iou_thr = 0.5
        # You can increase this during training to generate more propsals.
        RPN_NMS_THRESHOLD = 0.3
        RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        POST_NMS_ROIS_TRAINING = 2000

    #%% stable args
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    BACKBONE = "resnet50" #TODO
    
    # Image mean (RGB)
#    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL = np.array([218.37592,213.07745,211.30586])
    # Length of square anchor side in pixels
#   RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
#    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
#    RPN_ANCHOR_SCALES = (16, 32, 32, 64, 128)
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
#    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.
    }
############################################################
#  Dataset
############################################################
def drew_detect_resualt(path, save_path, bboxes, category_ids, class_info, debug=False):
    
    img = Image.open(path)
    if len(category_ids)==0:
#        print("image: no bbox", path)
        return
    ttfont = ImageFont.truetype('lib/华文细黑.ttf', 10)
    for idx, bbox in enumerate(bboxes):
        
        category_id = category_ids[idx]
        class_name = class_info[category_id]['name']
        draw = ImageDraw.Draw(img)
        draw.line([(bbox[0],bbox[1]),
               (bbox[2],bbox[1]),
               (bbox[2],bbox[3]),
               (bbox[0],bbox[3]),
               (bbox[0],bbox[1])], width=3, fill='yellow')
#            print('class_nameclass_nameclass_nameclass_name',class_name)
#            unicode('杨','utf-8')
        draw.text((bbox[0],bbox[1]), class_name, 
                  fill=(255,0,0), font= ttfont)
#    if debug:
#        img.show()
        
    img.save(save_path)
    
    
class ThisDataset(utils.Dataset):

    
    def prepare_class(self, categories):
    
        class_num = self.config.NUM_CLASSES
            
        for i in range(class_num-1): #TODO
            self.add_class(self.config.NAME, 
                           categories[i]['id'], 
                           categories[i]['name'])
        return  
    
    def load_data(self, config, subset):
        """Load a subset of the Balloon dataset.
        annotations_path: file path of the annotation.
        class_path: file path of class
        image_dir: the dictionary of image
        subset: Subset to load: train or val or test
        class_ids: class IDs to load
        """
        
        # all   289222
        # train 209222
        # val   40000
        # test  40000
        self.config = config
        exc = [[160338, 160352]]
        # Train or validation or test dataset?
        assert subset in ["train", "val", "test"]

        # Load annotations
        # We mostly care about the x and y coordinates of each region
        annotations_all = pd.read_csv(config.annotations_path)
#        img_Data = h5py.File(img_path,'r')
#        mask_all = img_Data['b_']
#        img_all = img_Data['ih']
#        img_mean = np.transpose(img_Data['ih_mean'][:][:][:], (2,1,0))
        class_all = pd.read_csv(config.class_path)

        annotations = annotations_all[annotations_all["evaluation_status"] == subset]
#        annotations = annotations_all
        class_ids = list(set(annotations.category_label))
        if class_ids:
            image_ids = []
            for i in class_ids:
                temp = annotations[annotations["category_label"] == i]
                image_ids.extend(list(temp["image_id"]))
        else:
            image_ids = list(set(annotations.image_id))

        for i in class_ids:
            self.add_class("clothes", i, class_all[class_all["category_label"] == i].iloc[0].category_name)

        # Add images  this step should be optimized to avoid applying too much memory
        print("Loading image!")
        
        f = open('dataset_log.txt', 'w')
        time0 = time.time()
        counter = 0
        for index, annotation in annotations.iterrows():
            counter += 1
            image_id=annotation.image_id
            img_full_path = os.path.join(config.IMG_DIR, annotation.image_name)
            bbx_ = [annotation.x_1, annotation.y_1, annotation.x_2, annotation.y_2]
#            print(image_id)
            
#            bbx, class_ids = self.process_one_image(img_full_path, 
#                                                    bbx_,
#                                                    annotation.category_label)
            
            
            if 'need_check_per_image1' == 'need_check_per_image':
                try:
                    img = image.load_img(img_full_path)
                except FileNotFoundError:
#                    print(annotation.image_name)
                    f.writelines(str(index) + ' : ' + annotation.image_name + '\n')
                    continue
            
#            img_b = (np.transpose(img_all[image_id][:][:][:],(2,1,0))+img_mean)*255
#            img_0 = np.where(img_b > 0, img_b, 0)
#            img_1 = np.where(img_0 < 255, img_0, 255)
#            if False:
#                img_2 = Image.fromarray(img_1.astype(np.uint8))
#                img_2.show()
            
            
            if (index >= exc[0][0] and index <= exc[0][1]):
                continue
                
            self.add_image(
                    config.NAME,
                    image_id=image_id,
                    path=img_full_path,
                    width=annotation.width,
                    height=annotation.height,
                    category_ids = [annotation.category_label],
                    bboxes = [bbx_]
                )
            step=2000
            if counter % step == 0:
                rest_time = (time.time()-time0)*((len(annotations)-counter)/(step))
                print('----Adding the image:', counter, 
                      'rest time(sec) = ', rest_time)
                time0 = time.time()
            if counter >1000:
                break
        
        f.close()
        print('-----------loaded total image ----------------:', counter)
        
        
#        for idx, i in enumerate(image_ids):
#            if idx%100 ==0:
#                print("Loading img: {}".format(idx))
#            self.add_image(
#                "clothes",
#                image_id=i,
#                path=image_dir + annotations[annotations["image_id"]==i].iloc[0].image_name,
#                width=annotations[annotations["image_id"]==i].iloc[0].width,
#                height=annotations[annotations["image_id"]==i].iloc[0].height,
#                class_id = annotations[annotations["image_id"]==i].iloc[0].category_label,
#            )
        
        
    def load_retrival_data(self, config):
        """
        Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        
        self.config = config
#        exc = [[77633, 77695], [160338, 160352]]
        annotations_all = pd.read_csv(config.annotations_path)
        class_all = pd.read_csv(config.class_path)
        if config.Mode == 'evaluate':
            annotations = annotations_all[annotations_all["evaluation_status"] == 'val']
        if config.Mode == 'retrival':
            annotations = annotations_all[annotations_all["evaluation_status"] == 'val']
        
        class_ids = list(set(annotations.category_label))
        if class_ids:
            image_ids = []
            for i in class_ids:
                temp = annotations[annotations["category_label"] == i]
                image_ids.extend(list(temp["image_id"]))
        else:
            image_ids = list(set(annotations.image_id))
        for i in class_ids:
            self.add_class("clothes", i, class_all[class_all["category_label"] == i].iloc[0].category_name)
        
        
        
        if config.Mode == 'evaluate':
            dataset_dir = config.DEFAULT_DATASET #'mini_subset' or 'subset'
           
            # All images or a subset?
#            files_list = sorted(os.listdir(dataset_dir))
            files_list = open(dataset_dir, 'r').readlines()
#            i = 0
#            for file in files_list:
#                if file[-4:]=='.jpg':
#                    image_ids.append(i)
#                    image_path_list.append(os.path.join(dataset_dir, file))
#                    i += 1
#                if i%100 == 0:
#                    print('Loading evaluate image_path:{}/{} '.format(i, len(files_list)))

            dataset_img_fold = os.path.join(os.path.split(dataset_dir)[0], 'subset')
            # Add images
            for i, img_info in enumerate(files_list):
                [img_id, img_name, w,h,c] = img_info.strip().split()
#                img = image.load_img(image_path_list[i])
#                img = np.array(img)
                self.add_image(
                    "clothes", 
                    image_id=str(img_id),
                    path=os.path.join(dataset_img_fold, img_name),
                    width=str(w),
                    height=str(h),
                    category_ids = 1)
                if i%1000 == 0:
                    print('Loading evaluate image_infi:{}/{} '.format(i, len(image_ids)))
                
        if config.Mode=='retrival':
            query_file = config.query_file +'/groundtruth.txt'
            
            query_infos = open(query_file, 'r').readlines()
            # Load all classes or a subset?
            # All classes
            class_ids = [i for i in range(2)]
            # Add classes
            for i in class_ids:
                self.add_class("clothes", i, 'class_'+str(i))
    
    
            image_ids = []
            for i, query_info in enumerate(query_infos):
                temp = query_info.strip().split(' ')
                image_path = os.path.join(config.query_file, temp[0])
                img = image.load_img(image_path)
                img = np.array(img)
                bbox = [float(temp[1]), 
                        float(temp[2]),
                        float(temp[3]),
                        float(temp[4])]
                
                # Add images
                self.add_image(
                    "clothes", 
                    image_id=i,
                    path=image_path,
                    bboxes = [bbox],
                    width=img.shape[1],
                    height=img.shape[0],
                    category_ids = [1],
                    related_num = int(temp[5]))
        
        
    def load_data_(self, config, img_dir, debug=False):
        """Load a subset of the Balloon dataset.
        annotations_path: file path of the annotation.
        class_path: file path of class
        image_dir: the dictionary of image
        subset: Subset to load: train or val or test
        class_ids: class IDs to load
        """
        f = open(config.annotion_path, encoding='utf-8')
        dataset = json.load(f)
        f.close()
        self.config = config
        self.prepare_class(dataset['categories'])
        
        # Add images  this step should be optimized to avoid applying too much memory
        print("Loading image!")
        json_path_list = scan_specified_files(img_dir, key='.json')
        
        f = open('dataset_log.txt', 'w')
        time0 = time.time()
        counter = 0
        image_id_repeat = 0
        for idx, json_full_path in enumerate(json_path_list):
            jf = open(json_full_path, encoding='utf-8')
            info = json.load(jf)
            jf.close()

            width=info['width']
            height=info['height']
            img_full_path = os.path.join(os.path.split(json_full_path)[0], info['file_name'])

            if 'need_check_per_image' == 'need_check_per_image':
                try:
                    img = image.load_img(img_full_path)
                except FileNotFoundError as e:
                    print(e)
#                    print(annotation.image_name)
                    f.writelines(str(idx) + ' : ' + img_full_path + '\n')
                    continue
                width_gt, height_gt = img.size  #TODO
                if [width, height] != [width_gt, height_gt]:
                    print('wrong width and height')
                    f.writelines(str(idx) + ': wrong width and height: '+img_full_path+'\n')
                    sys.exit()
                    continue
            
            re_category_ids = []
            re_bboxes = []
            if len(info['objects'])==0 and not config.USING_NEGATIVE_IMG:
#                print('ignore no NEGATIVE image')
                continue
            if len(info['objects'])> 0 and not config.USING_POSITIVE_IMG:
#                 print('ignore no POSITIVE image')
                continue
            
            for idx_, obj in enumerate(info['objects']):
                bbox = obj['bbox']
                
                x1 = min(bbox[0], bbox[2])
                y1 = min(bbox[1], bbox[3])
                x2 = max(bbox[0], bbox[2])
                y2 = max(bbox[1], bbox[3])
                if x1 >= x2 or y1 >= y2:
                    print('bbox_gt error ',bbox )
                    continue
                re_category_ids.append(obj['label'])
                rect = []
                rect.append(x1)
                rect.append(y1)
                rect.append(x2)
                rect.append(y2)
                re_bboxes.append(rect)
                
            if debug:
                save_path = 'train_data_virsual_fold'
                check_path(save_path)
                drew_detect_resualt(img_full_path, 
                                    os.path.join(save_path, img_full_path.split('/')[-1]), 
                                    re_bboxes, 
                                    re_category_ids, 
                                    self.class_info, 
                                    debug)
            
#            img_b = (np.transpose(img_all[image_id][:][:][:],(2,1,0))+img_mean)*255
#            img_0 = np.where(img_b > 0, img_b, 0)
#            img_1 = np.where(img_0 < 255, img_0, 255)
#            if False:
#                img_2 = Image.fromarray(img_1.astype(np.uint8))
#                img_2.show()
            
            repeat = 1
            if len(info['objects']) == 0:
                repeat = config.NEGATIVE_MULT
            if len(info['objects']) > 0:
                repeat = config.POSITIVE_MULT
            for i in range(repeat):
                self.add_image(
                        config.NAME,
                        image_id=image_id_repeat,
                        path=img_full_path,
                        width=width,
                        height=height,
                        category_ids = re_category_ids,
                        bboxes = re_bboxes
                    )
                image_id_repeat += 1
            counter += 1
            step=200
            if counter % step == 0:
                rest_time = (time.time()-time0)*((len(json_path_list)-counter)/(step))
                print('----Adding the image:', counter, 
                      'rest time(sec) = ', rest_time)
                time0 = time.time()
#            if counter >10:      #TODO
#                break

            
        f.close()
        print('-----------loaded total image ----------------:', counter)
        print('-----------after balance total----------------:', image_id_repeat)

    
    def load_data_only_image(self, config):
        """Load a subset of the Balloon dataset.
        annotations_path: file path of the annotation.
        class_path: file path of class
        image_dir: the dictionary of image
        subset: Subset to load: train or val or test
        class_ids: class IDs to load
        """
        
        
        
        f = open(config.annotion_path, encoding='utf-8')
        dataset = json.load(f)
        
        self.config = config

        self.prepare_class(dataset['categories'])
            
        test_img_dir = config.real_test_img_dir
        
        self.config = config
        
        # Add images  this step should be optimized to avoid applying too much memory
        print("Loading image!")
        
        img_path_list = scan_specified_files(test_img_dir, key='jpg')
        
        f = open('dataset_log.txt', 'w')
        time0 = time.time()
        counter = 0
        for image_id, img_full_path in enumerate(img_path_list):

            

            if 'need_check_per_image' == 'need_check_per_image':
                try:
                    img = image.load_img(img_full_path)
                except FileNotFoundError:
#                    print(annotation.image_name)
                    f.writelines(str(image_id) + ' : ' + img_full_path + '\n')
                    continue
            width, height = img.size  #TODO
#            img_b = (np.transpose(img_all[image_id][:][:][:],(2,1,0))+img_mean)*255
#            img_0 = np.where(img_b > 0, img_b, 0)
#            img_1 = np.where(img_0 < 255, img_0, 255)
#            if False:
#                img_2 = Image.fromarray(img_1.astype(np.uint8))
#                img_2.show()
            
                
            self.add_image(
                    config.NAME,
                    image_id=image_id,
                    path=img_full_path,
                    width=width,
                    height=height
                )
            counter += 1
            step=2000
            if counter % step == 0:
                rest_time = (time.time()-time0)*((len(img_path_list)-counter)/(step))
                print('----Adding the image:', counter, 
                      'rest time(sec) = ', rest_time)
                time0 = time.time()
            if counter > config.EVA_LIMIT :
                break
        
        f.close()
        print('-----------loaded total image ----------------:', counter)
        
        
    def process_one_image(self, image_path, bbox_ori, class_id):
        """
        
        """
        def resize_bbox(bbx, scale, padding, crop):
            
            bbx = [i*scale for i in bbx]
            bbx[0]+= padding[1][0]
            bbx[2]+= padding[1][0]
            bbx[1]+= padding[0][0]
            bbx[3]+= padding[0][0]
            bbx = [int(i) for i in bbx]
            
            return bbx
            
        config = self.config

#        class_id = self.map_source_class_id( "clothes.{}".format(class_id))
        
        class_ids = np.array([class_id]).astype(np.int32)
        
        img_img = image.load_img(image_path)
        img = image.img_to_array(img_img)
        if img.shape[:2] != (300, 300):
            print('sa')
        img_resize, window, scale, padding, crop = utils.resize_image(
        img,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
        
        class_ids = np.array([class_id]).astype(np.int32)
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = resize_bbox(bbox_ori, scale, padding, crop)
        bbox = np.array([bbox]).astype(np.int32)
        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.

#        if bbox.shape[0]>1:
#        print("------------bbox num:", bbox.shape[0])
        
        return bbox, class_ids
        
        
    
    def get_bbox_from_mask(self, mask_ori, image, class_id):
        """
        """
        config = self.config
        mask = np.transpose(mask_ori[:][:][:],(2,1,0))
        mask_raw = utils.resize_mask(mask,1,0)
        
        mask = np.equal(mask_raw, 3)

#        class_id = self.map_source_class_id( "clothes.{}".format(class_id))
        
        mask = mask.astype(np.bool)
        class_ids = np.array([class_id]).astype(np.int32)
        
        
        image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding, crop)
        
        
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = utils.extract_bboxes(mask)
    
        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.

#        if bbox.shape[0]>1:
#        print("------------bbox num:", bbox.shape[0])
        
        return bbox, class_ids
        
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_info = self.image_info[image_id]
        image = image_info["ih"]

        return image    
        
    def load_image_from_path(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_info = self.image_info[image_id]
        
        path = image_info["path"]
        img = image.load_img(path)
        img = image.img_to_array(img)
        return img.astype(np.uint8)
    def load_image_path(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_info = self.image_info[image_id]
        
        path = image_info["path"]
        return path
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.config.NAME:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    def load_bboxes(self, image_id):
        """
        generate bbox from mask
        """
        image_info = self.image_info[image_id]
        bboxes = image_info["bboxes"]
        category_ids = image_info["category_ids"]
        return bboxes, category_ids
    
    def load_related_num(self, image_id):
        """
        generate bbox from mask
        """
        image_info = self.image_info[image_id]
        related_num = image_info["related_num"]
        return related_num
    
def prepare_dataset(config, debug=False):
    
    split_dataset(config)

        
    split_record = config.split_record_dir
    try:
        postive_train_index = []
        postive_val_index = []
        nagetive_train_index = []
        nagetive_val_index = []
        postive_index = read_index(os.path.join(split_record, 'postive_index.txt'))
        nagetive_index = read_index(os.path.join(split_record, 'nagetive_index.txt'))
        
        split_idx0 = int(len(postive_index)*(1.0 - config.VAL_DATA_RATE))
        split_idx1 = int(len(nagetive_index)*(1.0 - config.VAL_DATA_RATE))
        
        postive_train_index = postive_index[: split_idx0]
        postive_val_index = postive_index[split_idx0: ]
        nagetive_train_index = nagetive_index[: split_idx1]
        nagetive_val_index = nagetive_index[split_idx1: ]
    except FileNotFoundError as e:
        print(e)
        print('split_record file not fond in :', split_record)
    
    f = open(config.annotion_path, encoding='utf-8')
    dataset = json.load(f)
    annotations = dataset['annotations']
    annotations = parse_data(annotations)
    # Training dataset.
    dataset_train = ThisDataset()
    dataset_train.load_data(config,
                            postive_train_index,
                            nagetive_train_index, 
                            annotations, debug)
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = ThisDataset()
    dataset_val.load_data(config,
                          postive_val_index,
                          nagetive_val_index,
                          annotations, debug)
    dataset_val.prepare()
    
    return dataset_train, dataset_val


def split_dataset(config):
    
    split_record = config.split_record_dir
    
    txt_list = os.listdir(split_record)
    if len(txt_list)>0:
        print('Worning: split_record is already existed')
        return False
    
    f = open(config.annotion_path, encoding='utf-8')
    dataset = json.load(f)
    
    # distribute train and test data by slice index
    data_index = list(range(len(dataset['images'])))
    random.shuffle(data_index)
    
    normal_img_dir = config.normal_img_dir
    img_name_list = sorted(os.listdir(normal_img_dir))
    data_index2 = list(range(len(img_name_list)))
    random.shuffle(data_index2)   
    
    with open(os.path.join(split_record, 'postive_index.txt'), 'w') as f:
        for idx in data_index:
            f.writelines(str(idx)+'\n')
    with open(os.path.join(split_record, 'nagetive_index.txt'), 'w') as f:
        for idx in data_index2:
            f.writelines(str(idx)+'\n')  

    print('----------split end---------------')
    
    return True


if __name__ == '__main__':
    
    config = ThisConfig()
    dataset_train = ThisDataset()
    dataset_train.load_data(config, config.train_img_dir, debug=True)
    dataset_train.prepare()
    
    # Validation dataset
#    dataset_val = ThisDataset()
#    dataset_val.load_data(config, config.val_img_dir, debug=True)
#    dataset_val.prepare()
    
    print('----jinnan.load_data(config) over-----')
    
    
    
    

