"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
#import sys
import time
import numpy as np
#import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import pandas as pd

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".

#from pycocotools import mask as maskUtils

#import zipfile
#import urllib.request
#import shutil


#added by Aaron
import pickle
import keras.preprocessing.image as image
from PIL import Image,ImageDraw
from sklearn.preprocessing import normalize
#

# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
#
## Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
#
from tools import check_path
from mrcnn.utils import compute_ap
import ThisConfig

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
def check_fold(path):
    if not os.path.exists(path):
        os.mkdir(path)

############################################################
#  Configurations
############################################################



############################################################
#  Dataset
############################################################

class ClothDataset(utils.Dataset):
    def load_cloth(self, config, dataset_dir, query_dir, annotations_path, class_path, img_path):
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
        annotations_all = pd.read_csv(annotations_path)
        class_all = pd.read_csv(class_path)
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
            dataset_dir = dataset_dir #'mini_subset' or 'subset'
           
            # All images or a subset?
#            files_list = sorted(os.listdir(dataset_dir))
            files_list = open(dataset_dir, 'r').readlines()
            image_path_list = []
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
                    class_id = 1)
                if i%1000 == 0:
                    print('Loading evaluate image_infi:{}/{} '.format(i, len(image_ids)))
                
        if config.Mode=='retrival':
            query_file = query_dir +'/groundtruth.txt'
            
            query_infos = open(query_file, 'r').readlines()
            # Load all classes or a subset?
            # All classes
            class_ids = [i for i in range(2)]
            # Add classes
            for i in class_ids:
                self.add_class("clothes", i, 'class_'+str(i))
    
    
            image_ids = []
            image_path_list = []
            for i, query_info in enumerate(query_infos):
                temp = query_info.strip().split(' ')
                image_path = os.path.join(query_dir, temp[0])
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
                    bbx = bbox,
                    width=img.shape[1],
                    height=img.shape[0],
                    class_id = 1,
                    related_num = int(temp[5]))
            
    # The following two functions are from pycocotools with a few changes.
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
        

    
    def load_image_from_path(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_info = self.image_info[image_id]
        
        path = image_info["path"]
        img0 = image.load_img(path)
        img1 = image.img_to_array(img0)
        img2 = img1.astype(np.uint8)
        return img2
    
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
        if info["source"] == "clothes":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    def load_bbox(self, image_id):
        """
        generate bbox from mask
        """
        image_info = self.image_info[image_id]
        bbox = image_info["bbx"]
        class_id = image_info["class_id"]
        return bbox, class_id
    def load_related_num(self, image_id):
        """
        generate bbox from mask
        """
        image_info = self.image_info[image_id]
        related_num = image_info["related_num"]
        return related_num

    def load_given_bbox(self, image_id):
        """
        Load the specified bbox in and return a [x1, y1, x2, y2] in normlized Numpy array.
        
        """
        bbox = self.image_info[image_id]['bbx']
        
        return bbox

############################################################
#  COCO Evaluation
############################################################


def build_cloth_evaluate_results(dataset, image_ids, rois, class_ids, scores, rmac, feats):
    """
    Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    
    # If no results, return an empty list
    if rois is None:
        return []
    total_result = []
    bbox_results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            feat = feats[i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, config.NAME),
                "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                "score": score,
                "feat": feat
            }
            bbox_results.append(result)
    total_result.append(bbox_results)
    
    rmac = np.sum(feats, axis=0)
    rmac = normalize([rmac])[0]
    total_result.append(rmac)
    
    total_result.append(feats)
    
    return total_result

def build_cloth_retrival_results(dataset, image_ids, rois, class_ids, scores, rmac, feats, given_feat):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []
    total_result = []
    bbox_results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            feat = feats[i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, config.NAME),
                "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                "score": score,
                "feat": feat
            }
            bbox_results.append(result)
    total_result.append(bbox_results)
    
    rmac = np.sum(feats, axis=0)
    rmac = normalize([rmac])[0]
    total_result.append(rmac)
    
    total_result.append(feats)
    
    total_result.append(given_feat)
    
    return total_result





        
def save_one_evaluate_result(config, image_id, dataset, result):
        """
        result = {
                "image_id": image_id,
                "path":dataset.source_image_link(image_id),
                "category_id": dataset.get_source_class_id(class_id, "cloth"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))}
        results.append(result)
        """
        [box_results, rmac, feats] = result
        base_dir = os.path.join(config.save_base_dir, 'feats_db_'+config.subset)
        check_fold(base_dir)
        
        save_feat_db_path = os.path.join(base_dir, 'feats')
        check_fold(save_feat_db_path)
        
        
        
        path = dataset.source_image_link(image_id)
        save_rmac_full_name = os.path.join(save_feat_db_path, 'img_'+str(image_id)+'.pkl')
        pickle.dump([image_id, path, result], open(save_rmac_full_name,'wb'))
        

            
        #rmac
        
        
        img = Image.open(path)
        if len(box_results)==0:
            print("image: no bbox", image_id)
            
        for bbox in box_results:
            
            box = bbox['bbox']
            draw = ImageDraw.Draw(img)
            draw.line([(box[0],box[1]),
                   (box[2],box[1]),
                   (box[2],box[3]),
                   (box[0],box[3]),
                   (box[0],box[1])], width=1, fill='yellow')
        #img.show()
        save_img_path = os.path.join(base_dir, 'images')
        check_fold(save_img_path)
        img.save(os.path.join(save_img_path, path.split('/')[-1]))

def save_one_retrival_result(config, image_id, dataset, result, related_num):
        """
        result = {
                "image_id": image_id,
                "path":dataset.source_image_link(image_id),
                "category_id": dataset.get_source_class_id(class_id, "cloth"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))}
        results.append(result)
        """
        [box_results, rmac, feats, given_feat] = result
        
        base_dir = os.path.join(config.save_base_dir, 'query_db')
        check_fold(base_dir)
        
        save_feat_query_path = os.path.join(base_dir, 'feats')
        check_fold(save_feat_query_path)
        path = dataset.source_image_link(image_id)
        query_box, _ = dataset.load_bboxes(image_id)
        query_box = query_box[0]
        save_rmac_full_name = os.path.join(save_feat_query_path, 'img_'+str(image_id)+'.pkl')
        pickle.dump([image_id, path, query_box, result, related_num], open(save_rmac_full_name,'wb'))
        
        
        
        
        
        path = dataset.source_image_link(image_id)
        img = Image.open(path)
        if len(box_results)==0:
            print("image: no bbox", image_id)
            
        for bbox in box_results:
            
            box = bbox['bbox']
            draw = ImageDraw.Draw(img)
            draw.line([(box[0],box[1]),
                   (box[2],box[1]),
                   (box[2],box[3]),
                   (box[0],box[3]),
                   (box[0],box[1])], width=1, fill='yellow')
        #img.show()
        save_img_path = os.path.join(base_dir, 'images')
        check_fold(save_img_path)
        img.save(os.path.join(save_img_path, path.split('/')[-1]))

def evaluate_task(config, model, dataset, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """

        
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    image_ids = image_ids[:config.EVA_LIMIT]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
    
    
    image = dataset.load_image_from_path(image_ids[0])
    bbox, _ = dataset.load_bboxes(image_ids[0])
    print('Try predict one image......')
    r_ = model.detect([image], verbose=0, given_rois = [bbox])
    r_ = r_
    print('Predicted one image.')

    t_prediction = 0
    t_start = time.time()
    t0 = time.time()
    counter = 0
    results = []
    
    for i, image_id in enumerate(image_ids):
        # Load image
        counter += 1
        image = dataset.load_image_from_path(image_id)
        bboxes, _ = dataset.load_bboxes(image_id)
        related_num = dataset.load_related_num(image_id)
        print('Processing image {}'.format(i))
        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0, given_rois=[bboxes])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        for j in range(len(r["class_ids"])):
            r["class_ids"][j] = 1
        image_results = build_cloth_retrival_results(dataset, [coco_image_ids[i]],
                                           r["rois"], 
                                           r["class_ids"],
                                           r["scores"],
                                           r["rmac"],
                                           r['feats'],
                                           r['given_feat'])
        save_one_retrival_result(config, image_id, dataset, image_results, related_num)
        results.extend(image_results)
        step = 1000
        if counter%step == 0:
            t1 = time.time()
            rest_time = (t1-t0)*(len(image_ids)-counter)/step
            print('processing image:{}  rest time(sec)={}'.format(counter, rest_time))
            t0 = time.time()



    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

def true_evaluate_task(config, model, dataset, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """

    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    image_ids = image_ids[:config.EVA_LIMIT]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
    
    image_ = dataset.load_image_from_path(image_ids[0])
    print('Try testing predict one image......')
    r_ = model.detect([image_], verbose=0)
    r_ = r_
    print('Predicted one image.')

    t_prediction = 0
    t_start = time.time()
    t0 = time.time()
    counter = 0
    results = []
    kpi = {'maps':[], 'precisions':[], 'recalls':[]}
    if os.path.exists(os.path.join(config.save_base_dir, 'results_tab.csv')):
        os.remove(os.path.join(config.save_base_dir, 'results_tab.csv'))
    for i, image_id in enumerate(image_ids):
        counter += 1
        
        # Load image
        image = dataset.load_image_from_path(image_id)
        print('Processing image {}'.format(i))
        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Cast masks to uint8 because COCO tools errors out on bool
        for j in range(len(r["class_ids"])):
            r["class_ids"][j] = 1
        image_results = build_cloth_evaluate_results(dataset, 
                                                     [coco_image_ids[i]],
                                                     r["rois"], 
                                                       r["class_ids"],
                                                       r["scores"],
                                                       r["rmac"],
                                                       r['feats'])
        save_one_evaluate_result(config, image_id, dataset, image_results)
        results.extend(image_results)
        step = 1000
        if counter%step == 0:
            t1 = time.time()
            rest_time = (t1-t0)*(len(image_ids)-counter)/step
            print('processing image:{}  rest time(sec)={}'.format(counter, rest_time))
            t0 = time.time()

    



    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
                




############################################################
#  Training
############################################################


if __name__ == '__main__':
    
    config = ThisConfig.ThisConfig()
    config.display()
    print("Command: ", config.Mode)


    # Create model
    if config.Mode == "retrival":
        model = modellib.MaskRCNN(mode="retrival", config=config,
                                  model_dir=config.DEFAULT_LOGS_DIR)
    elif config.Mode == "evaluate":
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=config.DEFAULT_LOGS_DIR)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(config.Mode))

    # Load weights
    if config.init_with == "this":
        model.load_weights(config.THIS_WEIGHT_PATH, by_name=True)
    elif config.init_with == "coco":
        model.load_weights(config.COCO_WEIGHTS_PATH, by_name=True)
    elif config.init_with == "last":
    # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

 # Validation dataset
    dataset_test = ThisConfig.ThisDataset()
    dataset_test.load_retrival_data(config)
    dataset_test.prepare()

    if config.Mode == "evaluate":
        true_evaluate_task(config, model, dataset_test)
    elif config.Mode == "retrival":
        print("Running Cloth evaluation on {} images.".format(config.EVA_LIMIT))
        evaluate_task(config, model, dataset_test)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(config.Mode))
