#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:51:56 2018

@author: aaron
"""

import os
import shutil

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def scan_specified_files(base_dir, sorted=True, key=None):
    
    file_path_list = []
    
    for root, dirs, files in os.walk(base_dir):
        
        for file in files:
            if key:
                if file.endswith(key):
#                    print(os.path.join(root,file))
                    file_path_list.append(os.path.join(root,file))
                    continue
                else:
                    continue
            file_path_list.append(os.path.join(root,file))
        
    return file_path_list
    


def task_rename_files(path1, path2):
    
    temp = os.listdir(path1)
    
    for i in range(len(temp)):
        ori_full_path = os.path.join(path1, 'img_'+str(i)+'.pkl')
        
        new_full_path = os.path.join(path2, 'img_'+str(i+20000)+'.pkl')
        
        shutil.copy(ori_full_path, new_full_path)
        
def compare_two_list(list1, list2):
    info = {'rest':[], 'over':[]}
    for file in list1:
        if file in list2:
            info['rest'].append(file)
        else:
            info['over'].append(file)
    return info

def compare_two_fold(fold1, fold2, changf1=lambda x:x, changf2=lambda x:x):

    changedlist1 = [changf1(x) for x in sorted(os.listdir(fold1))]
    changedlist2 = [changf2(x) for x in sorted(os.listdir(fold2))]
    
    info1 = compare_two_list(changedlist1, changedlist2)
    info2 = compare_two_list(changedlist2, changedlist1)
    
    return info1, info2
        
if __name__ == '__main__':
        
    path = '/hdisk/Ubuntu/backup/aWS/obj-detection/Mask_RCNN-master/samples/coco/rmac/feats_db_miniset/feats'
    
    new_path = '/hdisk/Ubuntu/backup/aWS/obj-detection/Mask_RCNN-master/samples/coco/rmac/feats_db_miniset/feats_rename'
    check_path(new_path)
    task_rename_files(path, new_path)

