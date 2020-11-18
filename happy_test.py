




import os
import sys
import math
import cv2
import matplotlib
import pylab as plt
import numpy as np
import util

from numpy import ma
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)

new_path = r'../tf_netbuilder'
sys.path.append(new_path)


from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions

register_tf_netbuilder_extensions()

thre1 = 0.1

from models.openpose_singlenet import create_openpose_singlenet
# Uncomment the relevant section: 1) pretrained model OR 2) model with loaded checkpoint
# model = create_openpose_singlenet(pretrained=True)
model = create_openpose_singlenet(pretrained=False)
weights_path = "output_singlenet/openpose_singlenet" # weights trained from scratch
model.load_weights(weights_path)




path='C:/Users/ps/Desktop/djz/datasets/fly_2017_dataset/val2017/'
# path_write='C:/Users/ps/Desktop/djz/datasets/fly_2017_dataset/val_output/'
path_write=r'C:\Users\ps\Desktop\djz\fly_tensorflow_Realtime_Multi-Person_Pose_Estimation\tf_lite\a/'
read_name=os.listdir(path)
number_of_pic=len(read_name)

import time
time_strat=time.time()
M,N = 6,6
# for K_0 in range(int(M*N)):
for K_0 in range(number_of_pic):
    name_of_pic = read_name[K_0]
    oriImg = cv2.imread(path+name_of_pic)  # B,G,R order
    input_img = cv2.resize(oriImg, (224, 224))
    input_img = input_img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)
    output_blobs = model.predict(inputs)
    visual =0
    if visual:
        heatmap = output_blobs[3]
        heatmap = np.squeeze(heatmap)  # output 1 is heatmaps
        heatmap_avg = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        plt.figure()
        plt.subplot(M, N, K_0 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(oriImg[:, :, [2, 1, 0]], 'gray')
        plt.imshow(heatmap_avg[:, :, 0], alpha=.5)
    else:
        heatmap_head = output_blobs[3][0, :, :, 0]
        # np.argmax(heatmap_head)
        # row = np.argmax(heatmap_head) // heatmap_head.shape[1]
        # col = np.argmax(heatmap_head) % heatmap_head.shape[1]

        heatmap_avg = cv2.resize(heatmap_head,
                                 (oriImg.shape[1], oriImg.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
        x,y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
        center = y, x
        radius = 3  # int(radius)
        cv2.circle(oriImg, center, radius, (255, 0, 0), 2)
        plt.imsave(path_write + name_of_pic, oriImg)
        # fly_output=oriImg+heatmap_avg

time_end=time.time()
print(time_end-time_strat)


#
# # Load a sample image
# test_image = 'resources/ski_224.jpg'
# test_image = 'resources/fly4.jpg'
# oriImg = cv2.imread(test_image) # B,G,R order
# plt.imshow(oriImg[:,:,[2,1,0]])
#
# input_img=cv2.resize(oriImg,(224,224))
# input_img = input_img[np.newaxis, :, :, [2, 1, 0]]
# inputs = tf.convert_to_tensor(input_img)
# print(inputs.shape)
# output_blobs = model.predict(inputs)
#
# paf = output_blobs[2]
# heatmap = output_blobs[3]
#
# print("Output shape (heatmap): " + str(heatmap.shape))
# print("Output shape (paf): " + str(paf.shape))
#
# heatmap = np.squeeze(heatmap)  # output 1 is heatmaps
# heatmap_avg = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
#
# paf = np.squeeze(paf)  # output 0 is PAFs
# paf_avg = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

# figure = plt.figure(figsize=(10, 10))
#
# plt.subplot(2, 1, 1, title='paf')
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# plt.imshow(oriImg[:, :, [2, 1, 0]])
# plt.imshow(paf_avg[:, :, 0], alpha=.5)
#
# # plt.subplot(2, 1, 2, title='heatmap')
# plt.figure()
# plt.xticks([])
# plt.yticks([])
# plt.grid(False)
# # plt.imshow(oriImg[:, :, [2, 1, 0]])
# plt.imshow(heatmap_avg[:, :, 0], alpha=.5)




# path ='./train/'
# read_name=os.listdir(path)
# number_of_pic=len(read_name)
# for K_0 in range(number_of_pic):
#     name_of_pic=read_name[K_0]
#     img=cv2.imread(path+name_of_pic,0)















