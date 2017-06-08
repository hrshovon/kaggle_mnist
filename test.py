import matplotlib.pyplot as plt
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from neuralnet_conv import cnn_model
from tqdm import tqdm
import random
IMG_SIZE=28
LR=1e-3
MODEL_NAME='mnist-{}-{}.model'.format(LR,'6conv')
test_data=np.load("test.csv.npy")
fig = plt.figure()
model=cnn_model(IMG_SIZE,LR)
model.load(MODEL_NAME)

#def interpret_one_hot_output():
#	return 
def show_some_pics():
	num=0
	st_index=random.randint(0,27000)
	for item in test_data[st_index:st_index+12]:
		img_data=item[0]
		#print(img_data)
		y=fig.add_subplot(3,4,num+1)
		orig=img_data
		data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
		model_out=model.predict([data])[0]
		str_label=np.argmax(model_out)
		#if np.argmax(model_out) == 1: str_label='Dog'
		#else: str_label = 'cat'
		y.imshow(orig,cmap='gray')
		plt.title(str_label)
		y.axes.get_xaxis().set_visible(False)
		y.axes.get_yaxis().set_visible(False)
		num+=1
    
	plt.show()
def test():
	num=0
	with open('submission-file.csv','w') as f:
		f.write('ImageId,Label\n')
		num=1
		for data in tqdm(test_data):
			img_data=data[0]
			orig=img_data
			data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
			model_out=model.predict([data])[0]
			f.write('{},{}\n'.format(num,np.argmax(model_out)))
			num+=1
show_some_pics()
