

####################### convert to tflite ###############################

import os
import importlib
import click
import tensorflow as tf
from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions
# from util import probe_model
from util import probe_model_singlenet
from models.openpose_singlenet import create_openpose_singlenet
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def export_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.experimental_new_converter = True
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(output_path, "wb").write(tflite_model)


register_tf_netbuilder_extensions()

# load saved model
module = importlib.import_module('models')
create_model_fn = 'create_openpose_singlenet'
create_model = getattr(module, create_model_fn)
path_weights = "output_singlenet/openpose_singlenet"
model = create_model()
model.load_weights(path_weights)

# first pass
probe_model_singlenet(model, test_img_path="resources/ski_224.jpg")

# export model to tflite

tflite_path = 'tf_lite/'

tflite_path = 'tf_lite/temp.tflite'
export_to_tflite(model, tflite_path)

print("Done !!!")



















## use tflite to predict

import os
import tensorflow as tf
import matplotlib.pylab as plt
import cv2
import numpy as np


tflite_model_file = 'tf_lite/temp.tflite'
#'openpose_singlenet.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
inp_index = interpreter.get_input_details()[0]["index"]
out_details = sorted(interpreter.get_output_details(), key=lambda k: k['index'])
heatmap_idx = out_details[-1]["index"]
paf_idx = out_details[-2]["index"]
print(interpreter.get_input_details())

path='C:/Users/ps/Desktop/djz/datasets/fly_2017_dataset/val2017/'
path_write='tf_lite/val_output/'
read_name=os.listdir(path)
number_of_pic=len(read_name)

import time


# time_strat=time.time()
M,N = 6,6

interpreter.allocate_tensors()
for K_0 in range(int(M*N)):
# for K_0 in range(number_of_pic):
#     K_0=1
#     time_strat=time.time()
    name_of_pic = read_name[K_0]
    oriImg = cv2.imread(path+name_of_pic)  # B,G,R order

    img = cv2.resize(oriImg, (224, 224))
    img = np.expand_dims(img, 0)

    input_tensor = tf.convert_to_tensor(img, np.uint8)
    # interpreter.allocate_tensors()
    interpreter.set_tensor(inp_index, input_tensor)

    time_strat = time.time()
    interpreter.invoke()

    time_end = time.time()
    print(time_end - time_strat)

    heatmaps = interpreter.get_tensor(heatmap_idx)
    # pafs = interpreter.get_tensor(paf_idx)


    # heatmap_idx = 0  # nose
    heatmap_head = heatmaps[0, :, :, 0]
    # plt.imshow(heatmap_head,'gray')
    heatmap_avg = cv2.resize(heatmap_head,
                             (oriImg.shape[1], oriImg.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
    # plt.imshow(heatmap_avg,'gray')
    x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
    center = y, x
    radius = 3  # int(radius)
    cv2.circle(oriImg, center, radius, (255, 0, 0), 2)
    # plt.imshow(oriImg,'gray')
    plt.imsave(path_write + name_of_pic, oriImg)
    time_end=time.time()
    print(time_end-time_strat)
# time_end=time.time()
# print(time_end-time_strat)











### new test
# import numpy as np
# import tensorflow as tf
# import cv2 as cv
#
# # Load TFLite model and allocate tensors.
# # tflite_model_file
# # tflite_model = tf.contrib.lite.Interpreter(model_path="/home/zhang/anaconda3/model_pb.tflite")
# tflite_model = tf.lite.Interpreter(model_path=tflite_model_file)
# tflite_model.allocate_tensors()
#
# # Get input and output tensors.
# input_details = tflite_model.get_input_details()
# output_details = tflite_model.get_output_details()
#
# # Test model on random input data.
#
# input_shape = input_details[0]['shape']
#
# image = cv.imread("resources/ski_224.jpg")
#
# # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # 输入随机数
#
# input_data = np.expand_dims(image, 0)
# tflite_model.set_tensor(input_details[0]['index'], input_data)
# # tflite_model.set_tensor(input_details[0]['index'], image)
#
#
# tflite_model.invoke()
# output_data = tflite_model.get_tensor(output_details[0]['index'])
# test_image = 'resources/fly4.jpg'
# img = cv2.imread(test_image) # B,G,R order
# oriImg = cv2.imread(test_image)  # B,G,R order
# img = cv2.resize(oriImg, (224, 224))
# img = np.expand_dims(img, 0)
#
# input_tensor= tf.convert_to_tensor(img, np.uint8)
# interpreter.allocate_tensors()
# interpreter.set_tensor(inp_index, input_tensor)
# interpreter.invoke()
# heatmaps = interpreter.get_tensor(heatmap_idx)
# pafs = interpreter.get_tensor(paf_idx)
#
# heatmap_idx = 0 # nose
# plt.imshow(heatmaps[0, :, :, heatmap_idx], cmap='gray')
#
# paf_dx_idx = 0
# paf_dy_idx = 1
# plt.imshow(pafs[0, :, :, paf_dx_idx], cmap='gray')
# plt.imshow(pafs[0, :, :, paf_dy_idx], cmap='gray')

# tflite_path = 'C:/Users/ps/Desktop/djz/' \
#               'fly_tensorflow_Realtime_Multi-Person_Pose_Estimation/resources/a'

# tflite_path = 'tf_lite/temp.tflite'
# open(tflite_path, "wb")

# @click.command()
# @click.option('--weights', required=True,
#               help='Path to the folder containing weights for the model')
# @click.option('--tflite-path',required=True,
#               help='Path to the output tflite file')
# @click.option('--create-model-fn',required=True,
#               help='Name of a function to create model instance.
#               Check available names here: .models._init__.py')
#
#
# def main(weights, tflite_path, create_model_fn):
#     register_tf_netbuilder_extensions()
#
#     # load saved model
#
#     module = importlib.import_module('models')
#     create_model = getattr(module, create_model_fn)
#
#     model = create_model()
#     model.load_weights(weights)
#
#     # first pass
#
#     probe_model_singlenet(model, test_img_path="resources/ski_224.jpg")
#
#     # export model to tflite
#
#     export_to_tflite(model, tflite_path)
#
#     print("Done !!!")
#
#
#
# if __name__ == '__main__':
#     main()

