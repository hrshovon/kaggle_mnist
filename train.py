import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import os
import tensorflow as tf
#from neuralnet_conv import cnn_model
from neuralnet_conv3 import cnn_model

IMGSIZE=28
LR=1e-4
MODEL_NAME='mnist-{}-{}.model'.format(LR,'8conv')
tf.reset_default_graph()

training_data=np.load('train.csv.npy')




def train():
    model=cnn_model(IMGSIZE,LR)
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded')
    train=training_data[:-500]
    test=training_data[-500:]

    X=np.array([i[0] for i in train]).reshape(-1,IMGSIZE,IMGSIZE,1)
    Y=[i[1] for i in train]

    test_x=np.array([i[0] for i in test]).reshape(-1,IMGSIZE,IMGSIZE,1)
    test_y=[i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)
train()
