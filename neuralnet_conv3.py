import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import os
import tensorflow as tf

def cnn_model(IMG_SIZE,LR):
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = local_response_normalization(convnet)	
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = local_response_normalization(convnet)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = local_response_normalization(convnet)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = local_response_normalization(convnet)
    
    convnet = fully_connected(convnet, 2000, activation='relu')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 10, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet,tensorboard_dir='log')
    return model

