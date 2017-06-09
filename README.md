An attempt to reach as high score as possible in kaggle mnist challenge.
Images are read from csv file,then deskewed and then stored in npy file.
A convolutional neural network is then used to classify them.
I havent used pandas before so I recorded the data from train and test file in my own way.Will use pandas soon,I hope.
Thanks to opencv website and specially sentdex for his great tutorials on tflearn. The neural network in the neuralnet_conv.py is actually a slightly modified version of his tutorial on youtube.

Update

Currently highest accuracy after deskew operation is 99.071%.Changed learning rate
and number of epochs
