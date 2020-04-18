# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:31:47 2020

@author: reza
"""
#%%
import os
import numpy as np
import pandas as pd

#%% 
##### Normal case directories
norm_folders = ['Norm0', 'Norm1', 'Norm2']
abnorm_folders =  ['Abnorm', 'Covid19', 'PNEUMONIA']
norm_dirs = [os.path.join('UoGeneva/imageset', folder) for folder in norm_folders]
abnorm_dirs = [os.path.join('UoGeneva/imageset', folder) for folder in abnorm_folders]

image_names_list = []
image_paths_list = []
image_label_list = []
labels_ind_list = []
for folder, directory in zip(norm_folders, norm_dirs):
    img_names = os.listdir(directory)
    img_paths = [os.path.join(directory, img_name) for img_name in img_names]
    num_imgs = len(img_names)
    labels = np.repeat(['Normal'], num_imgs)
    labels_ind = np.repeat([0], num_imgs)
    image_names_list = image_names_list + img_names
    image_paths_list = image_paths_list + img_paths
    image_label_list = image_label_list + list(labels)
    labels_ind_list = labels_ind_list + list(labels_ind)
    
for folder, directory in zip(abnorm_folders, abnorm_dirs):
    img_names = os.listdir(directory)
    img_paths = [os.path.join(directory, img_name) for img_name in img_names]
    num_imgs = len(img_names)
    labels = np.repeat(['Abnormal'], num_imgs)
    labels_ind = np.repeat([1], num_imgs)
    image_names_list = image_names_list + img_names
    image_paths_list = image_paths_list + img_paths
    image_label_list = image_label_list +list(labels)
    labels_ind_list = labels_ind_list + list(labels_ind)
    
abs_metadata_uog_df = pd.DataFrame({'Name':image_names_list , 'Path':image_paths_list, 'Label':image_label_list, 'Label_Ind':labels_ind_list})
abs_metadata_uog_df.to_csv('abs_metadata_uog.csv', index=False)