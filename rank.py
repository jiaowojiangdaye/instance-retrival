#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:46:54 2018

@author: aaron
"""

import os
import time
import pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from PIL import Image, ImageDraw

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)



def read_feats_db(feats_db_path):
    
    feats_list = os.listdir(feats_db_path)
    feats_db = {}
    image_ids = []
    paths = []
    rmacs = []
    feats = []
    bbox_infos = []
    for i in range(len(feats_list)):
        info_file =  os.path.join(feats_db_path, 'img_'+str(i)+'.pkl')
        
        with open(info_file ,'rb') as f:
            info_file = pickle.load(f)
            [image_id, image_path, result] = info_file
            image_ids.append(image_id)
            paths.append(image_path)
            [box_results, rmac, box_feats] = result
            rmacs.append(rmac)
            bbox_infos.append(box_results)

         
    rmacs = np.array(rmacs)
    feats = np.array(feats)
    feats_db['image_ids'] = image_ids
    feats_db['paths'] = paths
    feats_db['rmacs'] = rmacs
    feats_db['bbox_infos'] = bbox_infos
    return feats_db


def read_query_feats(path):
    
    query_list = os.listdir(path)
    query_db = {}
    image_ids = []
    paths = []
    query_boxes = []
    rmacs = []
    feats = []
    bboxs = []
    query_feats = []
    related_nums = []
    for i in range(len(query_list)):
        info_file =  os.path.join(path, 'img_'+str(i)+'.pkl')
#        print(i)
        with open(info_file ,'rb') as f:
            info_file = pickle.load(f)
            [image_id, image_path, given_box, result, related_num] = info_file
            image_ids.append(image_id)
            paths.append(image_path)
            query_boxes.append(given_box)
            [box_results, rmac, box_feats, given_feat] = result
            query_feats.append(given_feat)
            rmacs.append(rmac)
            related_nums.append(related_num)
            for j, bbox in enumerate(box_results):
                bboxs.append(bbox)
                feats.append(box_feats[j])
    
    rmacs = np.array(rmacs)
    feats = np.array(feats)
    query_feats = np.array(query_feats)
    query_db['image_ids'] = image_ids
    query_db['paths'] = paths
    query_db['rmacs'] =  rmacs
    query_db['rmacs'] =  feats
    query_db['bboxs'] = bboxs
    query_db['query_feats'] = query_feats
    query_db['query_boxes'] = query_boxes
    query_db['related_num'] = related_nums
    return query_db


def mergeImages(name, files, box=None, size=(224,224), axis=0):
    """
    notice that the gt bbox is not the same order as regions in coridinates
    
    """
    baseimg=Image.open(files[0])
    if box:
        box[0] = box[0]
        box[2] = box[2]
        box[1] = box[1]
        box[3] = box[3]
    
    
        draw = ImageDraw.Draw(baseimg)
        draw.line([(box[0],box[1]),
                       (box[2],box[1]),
                       (box[2],box[3]),
                       (box[0],box[3]),
                       (box[0],box[1])], width=8, fill='yellow')
    
    baseimg=baseimg.resize(size,Image.ANTIALIAS)
    basemat=np.atleast_2d(baseimg)
    for idx, file in enumerate(files[1:]):
        
        im=Image.open(file)
        #resize to same width
        im=im.resize(size,Image.ANTIALIAS)
        
        mat=np.atleast_2d(im)
        basemat=np.append(basemat,mat,axis=axis)
    report_img=Image.fromarray(basemat)
    report_img.save(name)

def mergeImages2(name, files, bboxes, size=(224,224), axis=0):
    """
    notice that the gt bbox is not the same order as regions in coridinates
    
    """


    baseimg=Image.open(files[0])
    box = bboxes[0]
    
    draw = ImageDraw.Draw(baseimg)
    draw.line([(box[0],box[1]),
                   (box[2],box[1]),
                   (box[2],box[3]),
                   (box[0],box[3]),
                   (box[0],box[1])], width=8, fill='yellow')
    
    baseimg=baseimg.resize(size,Image.ANTIALIAS)
    basemat=np.atleast_2d(baseimg)
    bboxes = bboxes[1:]
    for idx, file in enumerate(files[1:]):
        box = bboxes[idx]
        im=Image.open(file)
        #resize to same width
        draw = ImageDraw.Draw(im)
        draw.line([(box[0],box[1]),
                   (box[2],box[1]),
                   (box[2],box[3]),
                   (box[0],box[3]),
                   (box[0],box[1])], width=8, fill='red')
        
        im=im.resize(size,Image.ANTIALIAS)
        
        mat=np.atleast_2d(im)
        basemat=np.append(basemat,mat,axis=axis)
    report_img=Image.fromarray(basemat)
    report_img.save(name)






class Ranker():
    
    def __init__(self, mark, base):

        # Read image lists
        self.dimension = 256
        
        self.rank_top_k = 200
        self.top_k = 10
        self.feats_db_path = base+ '/feats_db_'+mark+'/feats'
        query_feats_path = base+ '/query_db/feats'
        
        # Distance type
        self.dist_type = 'cosine'
        # Load features
        self.query_db = read_query_feats(query_feats_path)
        

        # Where to store the rankings
        self.rankings_dir = base+ '/rank_'+mark
        check_dir(self.rankings_dir)
        self.rerankings_dir = base+ '/rerank_'+mark
        check_dir(self.rerankings_dir)
    def load_db_feats(self):
        
        self.db_feats = read_feats_db(self.feats_db_path)
        
        
    def get_distances(self):
        
        db_feats = self.db_feats['rmacs']
        query_feats = self.query_db['query_feats']
        
        
        distances = pairwise_distances(query_feats,db_feats,self.dist_type, n_jobs=-1)

        return distances


    def write_rankings(self, final_scores):

        save_rank_path = os.path.join(self.rankings_dir, 'rank_order')
        check_dir(save_rank_path)
        
        for i, query_info in enumerate(self.query_db['query_feats']):

            scores = final_scores[i,:]

            ranking = np.arange(len(self.db_feats['rmacs']))[np.argsort(scores)]
            
            savefile = open(os.path.join(save_rank_path, 'rank_'+str(i)) +'.txt','w')

            for res in ranking:
                savefile.write(str(res) +': ' + self.db_feats['paths'][res] + '\n')
                
            savefile.close()
            
        


    def rank(self):
        
        print("Computing distances...")
        t0 = time.time()
        distances = self.get_distances()
        print( "Done. Time elapsed", time.time() - t0)

        print( "Writing rankings to disk...")
        t0 = time.time()
        self.write_rankings(distances)
        print( "Done. Time elapsed", time.time() - t0)
        
    def save_rank_result(self):
        save_image_path = os.path.join(self.rankings_dir, 'result_images')
        check_dir(save_image_path)
        
        save_rank_path = os.path.join(self.rankings_dir, 'rank_order')
        rank_files = os.listdir(save_rank_path)
        for i in range(len(rank_files)):
            ranks = open(os.path.join(save_rank_path, 'rank_'+str(i)+'.txt'),'r').readlines()
            
            #def mergeImages(name, files, box, size=(224,224), axis=0):
            files = []
            files.append(self.query_db['paths'][i])
            box = self.query_db['query_boxes'][i]
            name = os.path.join(save_image_path, 'result_'+str(i)+'.jpg')
            
            for i, order in enumerate(ranks):
                if i == self.top_k:
                    break
                image_path = order.strip().split(' ')[1]
                files.append(image_path)
            
            
            mergeImages(name, files, box, size=(128,128), axis=1)
            
            
        name_list = []
        name_all = os.path.join(save_image_path, '0000_10.jpg')
        
        for i in range(len(self.query_db['query_feats'])):
            
#            if i == 10:
#                break
            name_list.append(os.path.join(save_image_path, 'result_'+str(i)+'.jpg'))
            
        mergeImages(name_all, name_list, box=None, size=(128*(self.top_k+1),128), axis=0)

    def save_rank_kpi2(self):
        save_kpi_path = os.path.join(self.rankings_dir, 'result_kpi')
        check_dir(save_kpi_path)
        save_rank_path = os.path.join(self.rankings_dir, 'rank_order')
        rank_files = os.listdir(save_rank_path)
        
        with open(os.path.join(save_kpi_path, 'rank_kpi.txt'), 'a+') as f:
            
            releated_num_total = 0
            right_num_total = 0
            wrong_num_total = 0
            recall = 0
            precise = 0
            for i in range(len(rank_files)):
                ranks = open(os.path.join(save_rank_path, 'rank_'+str(i)+'.txt'),'r').readlines()
                
                #def mergeImages(name, files, box, size=(224,224), axis=0):
                files = []
                query_full_path = self.query_db['paths'][i]
                query_base_name = os.path.basename(query_full_path)
                query_base_name = query_base_name.split('.')[0]
                query_base_name = query_base_name[:-len(query_base_name.split('_')[-1])]
                
                files.append(query_full_path)
                
                related_num = self.query_db['related_num'][i]
                releated_num_total += related_num
                for i, order in enumerate(ranks[:self.top_k]):
                    
                    image_full_path = order.strip().split(' ')[1]
                    image_base_name = os.path.basename(image_full_path)
                    image_base_name = image_base_name.split('.')[0]
                    image_base_name = image_base_name[:-len(image_base_name.split('_')[-1])]
                    
                    if image_base_name == query_base_name:
                        right_num_total += 1
                    else:
                        wrong_num_total += 1
            try:
                recall = right_num_total*1.0/releated_num_total
                precise = right_num_total*1.0/(right_num_total + wrong_num_total)
            except ZeroDivisionError as e:
                print('--illeage value display_N')
            print('--top_k = : ', self.top_k)
            print('---- rank recall : ', recall)
            print('---- rank precise: ', precise)
            f.writelines('top_k = : '+str(self.top_k)+'\n')
            f.writelines('recall:  '+str(recall)[:6]+'\n')
            f.writelines('precise: '+str(precise)[:6]+'\n')
            f.writelines('-------------------------''\n')
            f.writelines('-------------------------''\n')
                
            
        return 

    def save_rank_kpi(self):
        save_kpi_path = os.path.join(self.rankings_dir, 'result_kpi')
        check_dir(save_kpi_path)
        save_rank_path = os.path.join(self.rankings_dir, 'rank_order')
        rank_files = os.listdir(save_rank_path)
        
        with open(os.path.join(save_kpi_path, 'rank_kpi.txt'), 'a+') as f:
            
            releated_num_total = 0
            right_num_total = 0
            wrong_num_total = 0
            recall = 0
            precise = 0
            for i in range(len(rank_files)):
                ranks = open(os.path.join(save_rank_path, 'rank_'+str(i)+'.txt'),'r').readlines()
                
                #def mergeImages(name, files, box, size=(224,224), axis=0):
                files = []
                query_full_path = self.query_db['paths'][i]
                query_base_name = os.path.basename(query_full_path)
                query_base_name = query_base_name.split('.')[0]
                query_base_name = query_base_name[:-len(query_base_name.split('_')[-1])]
                
                files.append(query_full_path)
                
                related_num = self.query_db['related_num'][i]
                releated_num_total += related_num
                for i, order in enumerate(ranks[: self.top_k]):
                    
                    image_full_path = order.strip().split(' ')[1]
                    image_base_name = os.path.basename(image_full_path)
                    image_base_name = image_base_name.split('.')[0]
                    image_base_name = image_base_name[:-len(image_base_name.split('_')[-1])]
                    
                    if image_base_name == query_base_name:
                        right_num_total += 1
                    else:
                        wrong_num_total += 1
            try:
                recall = right_num_total*1.0/releated_num_total
                precise = right_num_total*1.0/(right_num_total + wrong_num_total)
            except ZeroDivisionError as e:
                print('--illeage value top_k')
            print('--top_k = : ', self.top_k)
            print('---- rank recall : ', recall)
            print('---- rank precise: ', precise)
            f.writelines('top_k = : '+str(self.top_k)+'\n')
            f.writelines('recall:  '+str(recall)[:6]+'\n')
            f.writelines('precise: '+str(precise)[:6]+'\n')
            f.writelines('-------------------------''\n')
            f.writelines('-------------------------''\n')
                
            
        return 

    def read_rank(self, i):
        save_rank_path = os.path.join(self.rankings_dir, 'rank_order')
        rank_result_txt = os.path.join(save_rank_path, 'rank_'+str(i)+'.txt')
        f = open(rank_result_txt, 'r')
        ranks = f.readlines()[:self.top_k]
        result = []
        for line in ranks:
            temp = line.strip().split(' ')
            result.append([int(temp[0][:-1]), temp[1]])
        
        
        return result
        
        
    def rerank_once(self, query_idx, rank):
        print('rerank_once', query_idx)
        query_feat = self.query_db['query_feats'][query_idx]
        bboxes = []
        distances = []
        frames = []
        for one in rank:
            bbox_info = self.db_feats['bbox_infos'][one[0]]
            frames.append(one[1])
            bbox_feats = []
            for bbox in bbox_info:
                bbox_feats.append(bbox['feat'])
            bbox_feats = np.array(bbox_feats)
            if len(bbox_info) == 2:
#                print(' ')
                pass
            # Compute distances
            dist_array = pairwise_distances(query_feat.reshape(1, -1), bbox_feats, self.dist_type, n_jobs=-1)

            # Select minimum distance
            distances.append(np.min(dist_array))

            # Array of boxes with min distance
            idx = np.argmin(dist_array)

            # Select array of locations with minimum distance
            best_box_array = bbox_info[idx]['bbox']
            
            bboxes.append(best_box_array)
        
        
        dist = list(zip(distances, range(len(distances))))
        dist.sort(key = lambda x : x[0])
        best_idxs = [x[1] for x in dist[:self.top_k]]
        #choose the best 10 pictures
        best_boxes = list(map(lambda x: bboxes[x], best_idxs))
        best_distances = list(map(lambda x: distances[x], best_idxs))
#        frames
        best_frames = list(map(lambda x: frames[x], best_idxs))
        
        
        rerankings_dir_info = os.path.join(self.rerankings_dir, 'rerank_order')
        check_dir(rerankings_dir_info)
        with open(os.path.join(rerankings_dir_info, str(query_idx)) + '.pkl' ,'wb') as f:
            pickle.dump(best_distances, f)
            pickle.dump(best_boxes, f)
            pickle.dump(best_frames, f)
        
        
        return best_idxs, best_boxes, best_distances, best_frames
        
        
    def rerank(self):
        query_ids = self.query_db['image_ids']
        
        for idx, query_idx in enumerate(query_ids):
            rank = self.read_rank(idx)
            best_idxs, best_boxes, best_distances, best_frames = self.rerank_once(query_idx, rank)
            
            
            
            
    def save_rerank_result(self):
        '''
        
        
        '''
        
        save_image_path = os.path.join(self.rerankings_dir, 'result_images')
        check_dir(save_image_path)
        
        rerankings_dir_info = os.path.join(self.rerankings_dir, 'rerank_order')
        
        rerank_files = sorted(os.listdir(rerankings_dir_info))
        
        for i in range(len(rerank_files)):
            with open(os.path.join(rerankings_dir_info, str(i)+'.pkl') ,'rb') as f: 
                best_distances = pickle.load(f)
                best_boxes = pickle.load(f)
                best_frames = pickle.load(f)
            

            
            #def mergeImages(name, files, box, size=(224,224), axis=0):
            files = []
            files.append(self.query_db['paths'][i])
            bboxes = []
            box = self.query_db['query_boxes'][i]
            bboxes.append(box)
            name = os.path.join(save_image_path, 'result_'+str(i)+'.jpg')
            
            for i, frame in enumerate(best_frames):
                if i == self.top_k:
                    break
                files.append(frame)
                bboxes.append(best_boxes[i])
            
            mergeImages2(name, files, bboxes, size=(128,128), axis=1)
            
            
        name_list = []
        name_all = os.path.join(save_image_path, 'all.jpg')
        for i in range(len(rerank_files)):
            
#            if i == 10:
#                break
            name_list.append(os.path.join(save_image_path, 'result_'+str(i)+'.jpg'))
            
            
        mergeImages(name_all, name_list, box=None, size=(128*(self.top_k+1),128), axis=0)
        
        
        
        

                
if __name__=='__main__':
    
    mark = 'subset'  #'subset' 'miniset'
    base = 'clothes'
    version = '0742'
    ranker = Ranker(mark,'rmac_based_'+ base+'_'+version)
    
    ranker.load_db_feats()
    ranker.rank()
    
    ranker.save_rank_kpi2()
#    ranker.save_rank_result()
#    
#    ranker.rerank()
#    ranker.save_rerank_result()
    


