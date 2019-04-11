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
import json

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
#import pickle
#import keras.preprocessing.image as klimage
from PIL import Image,ImageDraw,ImageFont
#from sklearn.preprocessing import normalize
#

# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
#
## Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib
from ThisConfig import ThisConfig, ThisDataset, prepare_dataset
# Path to trained weights file
#
from tools import check_path
from mrcnn.utils import compute_ap


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs


############################################################
#  COCO Evaluation
############################################################

def get_avg_height(rois):
    heights = []
    weights = []
    for bbox in rois:
        weights.append(abs(bbox[3] - bbox[1]))
        heights.append(abs(bbox[2] - bbox[0]))
    avgw, avgh = 50, 50
    if len(rois) > 0:
        avgw = sum(weights)*1.0/len(weights)
        avgh = sum(heights)*1.0/len(heights)
    return avgw, avgh

def build_cloth_evaluate_results(dataset, image_ids, rois, class_ids, scores):
    """
    Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    def is_object(bbox, weight, height):
        thisw = abs(bbox[3] - bbox[1])
        thish = abs(bbox[2] - bbox[0])
        if abs(weight-thisw) < 0.55*weight and abs(height-thish) < 0.55*thish:
            return True
        else:
            return False
    
#    avg_weight, avg_height = get_avg_height(rois)
        
    
    # If no results, return an empty list
    if rois is None:
        return []
    total_result = []
    bbox_results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            bbox = np.around(rois[i], 1)
            
                
            class_id = class_ids[i]
            score = scores[i]
            

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, config.NAME),
                "bbox": [bbox[1], bbox[0], bbox[3], bbox[2]],
                "score": score,
            }
            bbox_results.append(result)
    total_result.append(bbox_results)
    
    return total_result

def get_standard_img_result(dataset, image_ids, rois, class_ids, scores):
    
    # If no results, return an empty list
    re = {}
    img_path = dataset.source_image_link(image_ids[0])
    re['filename'] = img_path.split('/')[-1]
    re['rects'] = []
    if rois is None:
        return re
    
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)

            rect = {"confidence": float(score),
                    "label": int(dataset.get_source_class_id(class_id, config.NAME)),
                    "xmin": int(bbox[1]),
                    "ymin": int(bbox[0]),
                    "xmax": int(bbox[3]),
                    "ymax": int(bbox[2])}
            re['rects'].append(rect)
            
    return re
    
def save_one_evaluate_result(config, image_id, dataset, result, avg_height, debug=False):
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
        [box_results] = result
        base_dir = config.save_base_dir
        check_path(base_dir)
        
        image_info_ = dataset.image_info[image_id]
        
        path = dataset.source_image_link(image_id)
        
        #%% detection 
        save_img_path = os.path.join(base_dir, 'detect_results')
        check_path(save_img_path)
        
        img = Image.open(path)
        gt_bboxes = []
        
        if 'bboxes' in image_info_.keys():
            gt_bboxes = image_info_['bboxes']
            gt_class_ids = image_info_['category_ids']
        
        if len(gt_bboxes)==0:
            pass
#            print("image: no bbox", image_id)
        text_size = int(20)
        ttfont = ImageFont.truetype('lib/华文细黑.ttf', text_size)
        for idx, box in enumerate(gt_bboxes):
            
            category_id = gt_class_ids[idx]
            class_name = dataset.class_info[category_id]['name']
            draw = ImageDraw.Draw(img)
            draw.line([(box[0],box[1]),
                   (box[2],box[1]),
                   (box[2],box[3]),
                   (box[0],box[3]),
                   (box[0],box[1])], width=3, fill='red')
#            print('class_nameclass_nameclass_nameclass_name',class_name)
#            unicode('杨','utf-8')
            draw.text((box[0]+10,box[1]), class_name.split('_')[0], 
                      fill=(255,0,0), font= ttfont)
        
        if len(box_results)==0:
            print("image: no bbox", image_id)
            
        pure_bboxes = []
        pure_class_ids = []
        for bbox in box_results:
            
            box = bbox['bbox']
            pure_bboxes.append(box)
            category_id = bbox['category_id']
            pure_class_ids.append(category_id)
            class_name = dataset.class_info[category_id]['name']
            draw = ImageDraw.Draw(img)
            draw.line([(box[0],box[1]),
                   (box[2],box[1]),
                   (box[2],box[3]),
                   (box[0],box[3]),
                   (box[0],box[1])], width=3, fill='blue')
            
            draw.text((box[0]+10,box[1]+text_size), class_name.split('_')[0],
                      fill=(0,0 ,255), font= ttfont)
            draw.text((box[0]+10,box[1]+text_size*2), str(bbox['score'])[:4],
                      fill=(0,0, 255), font= ttfont)
        if debug:
            img.show()
        img.save(save_img_path+'/'+path.split('/')[-1])
   
def compute_map(image_info, r, iou_threshold=0.5):
    
    
    gt_bboxes = []
    gt_class_ids = []
    for idx, bbox in enumerate(image_info['bboxes']):
        gt_bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])
        gt_class_ids.append(image_info['category_ids'][idx])
    gt_bboxes = np.array(gt_bboxes)
    gt_class_ids = np.array(gt_class_ids)
    mAP = 1
    precision = 1
    recall = 1
    if len(gt_bboxes.shape) == 2:
        mAP, precisions, recalls, overlaps = compute_ap(gt_bboxes, 
                                                        gt_class_ids,
                                                        r['rois'], 
                                                        r["class_ids"], 
                                                        r["scores"], 
                                                        iou_threshold=0.5)
    
        if len(precisions) >=3:
            precision = precisions[-2]
            recall = recalls[-2]
    else:
        print('gt error  no bboxes')
    
    return mAP, precision, recall

def save_kpi(config, kpi):
    avg_map = sum(kpi['maps'])*1.0/len(kpi['maps'])
    avg_pre = sum(kpi['precisions'])*1.0/len(kpi['precisions'])
    avg_rec = sum(kpi['recalls'])*1.0/len(kpi['recalls'])
    print('----------map={}-----------'.format(avg_map))
    print('----------precision={}-----------'.format(avg_pre))
    print('----------recall={}-----------'.format(avg_rec))
    base_dir = config.save_base_dir
    kpi_file = os.path.join(base_dir, 'kpi.txt')
    with open(kpi_file, 'a') as f:
        f.writelines('-----'+config.save_base_dir+'\n')
        f.writelines('avg_map = '+str(avg_map)+'\n')
        f.writelines('avg_precision = '+str(avg_map)+'\n')
        f.writelines('avg_recall = '+str(avg_map)+'\n')
    
    
    
    

def evaluate_task(config, model, dataset, limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """

    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

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
        image_results = build_cloth_evaluate_results(dataset, 
                                                     [coco_image_ids[i]],
                                                     r["rois"], 
                                                       r["class_ids"],
                                                       r["scores"])
        avg_weight, avg_height = get_avg_height(r["rois"])
        image_info_ = dataset.image_info[image_id]
        save_one_evaluate_result(config, image_id, dataset, image_results, avg_height)
        if not config.real_test:
            one_map, one_pre, one_recall = compute_map(image_info_, r, config.map_iou_thr)
            kpi['maps'].append(one_map)
            kpi['precisions'].append(one_pre)
            kpi['recalls'].append(one_recall)
        if config.real_test:
            one_re = get_standard_img_result(dataset, [coco_image_ids[i]], r["rois"], r["class_ids"], r["scores"])
            results.append(one_re)
        step = 1000
        if counter%step == 0:
            t1 = time.time()
            rest_time = (t1-t0)*(len(image_ids)-counter)/step
            print('processing image:{}  rest time(sec)={}'.format(counter, rest_time))
            t0 = time.time()
    if not config.real_test:
        save_kpi(config, kpi)
        
    if config.real_test:
        submit = {'results': results}
        submit_f = open(config.submit_path,'w',encoding='utf-8')
        json.dump(submit, submit_f)
        submit_f.close()
    

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    
    config = ThisConfig()
    config.display()
    

    # Create model
    if config.Mode == "evaluate":
        print("Loading weights ")
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=config.DEFAULT_LOGS_DIR)
        # Load weights
        if config.init_with == "this":
            model.load_weights(config.THIS_WEIGHT_PATH, by_name=True)
        elif config.init_with == "coco":
            model.load_weights(config.COCO_WEIGHTS_PATH, by_name=True)
        elif config.init_with == "last":
        # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)
        

        dataset_test = ThisDataset()
        if config.real_test:
            dataset_test.load_data_only_image(config)
        else:
            dataset_test.load_data(config, config.val_img_dir, debug=False)

        dataset_test.prepare()
        
        
        print("Running Cloth evaluation on {} images.".format(config.EVA_LIMIT))
        evaluate_task(config, model, dataset_test, limit=int(config.EVA_LIMIT))
        
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(config.Mode))
