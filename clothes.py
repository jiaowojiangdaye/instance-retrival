import os
import sys
import json
import datetime
import time
import h5py
import numpy as np
import pandas as pd
import skimage.io
from imgaug import augmenters as iaa
from keras.preprocessing import image
from PIL import Image
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
CUR_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(CUR_DIR, "logs")

############################################################
#  Configurations
############################################################


class ClothConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "clothes"

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 46 + 1 # Background + class

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 21
    
    VALIDATION_STEPS = 100
    
    IMAGE_MIN_DIM = 128
    
    IMAGE_MAX_DIM = 128
    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Image mean (RGB)
    MEAN_PIXEL = np.array([218.37592,213.07745,211.30586])
    
    # Skip detections with < 60% confidence
    DETECTION_MIN_CONFIDENCE = 0.1
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.3

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    
    BACKBONE = "resnet50"
    EPOCHS = 5000
############################################################
#  Dataset
############################################################

class ClothDataset(utils.Dataset):

    def load_cloth(self, config, annotations_path, class_path, img_path, mask_path, image_dir, subset, class_ids=None):
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
        annotations_all = pd.read_csv(annotations_path)
#        img_Data = h5py.File(img_path,'r')
#        mask_all = img_Data['b_']
#        img_all = img_Data['ih']
#        img_mean = np.transpose(img_Data['ih_mean'][:][:][:], (2,1,0))
        class_all = pd.read_csv(class_path)

        annotations = annotations_all[annotations_all["evaluation_status"] == subset]
#        annotations = annotations_all
        if not class_ids:
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
            img_full_path = os.path.join(image_dir, annotation.image_name)
            bbx_ = [annotation.x_1, annotation.y_1, annotation.x_2, annotation.y_2]
#            print(image_id)
            
#            bbx, class_ids = self.process_one_image(img_full_path, 
#                                                    bbx_,
#                                                    annotation.category_label)
            
            
            if 'need_check_per_image1' == 'need_check_per_image':
                try:
                    img = image.load_img(img_full_path)
                except FileNotFoundError as e:
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
                    class_id = [annotation.category_label],
                    bbx = [bbx_]
                )
            step=2000
            if counter % step == 0:
                rest_time = (time.time()-time0)*((len(annotations)-counter)/(step))
                print('----Adding the image:', counter, 
                      'rest time(sec) = ', rest_time)
                time0 = time.time()
#            if counter >1000:
#                break
        
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
        

