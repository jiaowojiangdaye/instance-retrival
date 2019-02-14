
# coding: utf-8

# Function: Annotation of  DeepFashion Dataset
# 
# Author: Jay Lee

# In[4]:


import os
import pandas as pd
import numpy as np
import skimage.io


# # 第一部分
# 
# 将原数据集中的服装图片各类信息汇总至一个csv文件中

# In[ ]:


# list_category_img.txt 文件中包含图片的服装种类标签
path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_category_img.txt"
data_1 = pd.read_table(path, header=1, sep="\s+")
# list_bbox.txt 文件中包含服装图片的边界框的信息 x1 y1 x2 y2
path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_bbox.txt"
data_2 = pd.read_table(path, header=1, sep="\s+")
# list_eval_partition.txt 文件中包含服装图片的数据集分类标签 train val test
path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Eval/list_eval_partition.txt"
data_3 = pd.read_table(path, header=1, sep="\s+")
# 合并三个文件
data = pd.merge(data_1, data_2, on=["image_name"])
data = pd.merge(data, data_3, on=["image_name"])
# 生成每幅图像的image_id
data["image_id"] = range(289222)
# 获取每幅图像的长和宽
data["height"] = None
data["width"] = None
img_dir = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Img/"

for i in range(289222):
    image_path = img_dir+data.iloc[i].image_name
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]
    data.loc[i, "height"] = height
    data.loc[i, "width"] = width
# 存储图像信息至 list_clothes.txt 文件中    
data.to_csv("/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_clothes.csv", index=False)


# # 第二部分
# 
# 从第一部分中去除具有掩码的服装图像

# In[3]:


# 读前部分生成的服装图像信息文件 list_clothes.csv
dataset_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_clothes.csv"
annotations_all = pd.read_csv(dataset_path)
print(annotations_all)


# In[2]:


# list_category_cloth.txt 文件中包含服装的类型信息 上衣 1 裤子等类 2 全身装 3
# 为了后面通过此标签读取相应的服装掩码
path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_category_cloth.txt"
data = pd.read_table(path, header=1, sep="\s+")
data["category_label"] = range(1,51)
print(data)
data.to_csv("/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_class.csv", index=False)


# In[4]:


# 将服装的类别信息加入到总的服装信息文件中
dataset_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_clothes.csv"
annotations_all = pd.read_csv(dataset_path)
label_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/list_class.csv"
label_all = pd.read_csv(label_path)
data = pd.merge(annotations_all, label_all, on=["category_label"])
save_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/cloth_all.csv"
data.to_csv(save_path, index=False)
print(data)


# In[7]:


# 取出具有掩码的服装图像的信息
index_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/clothes_subset_index.csv"
subset = pd.read_csv(index_path)
cloth_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/cloth_all.csv"
cloth_all = pd.read_csv(cloth_path)

data = pd.merge(subset, cloth_all, on=["image_name","image_id"])
data.image_id = range(78979)
save_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/cloth_subset.csv"
data.to_csv(save_path, index=False)xinx
print(data)


# In[9]:


data_path = "/home/mbzhao/Guests/Jay/Mask_RCNN-master/datasets/clothes/Anno/cloth_subset.csv"
data = pd.read_csv(data_path)
print(data.category_label.unique())

