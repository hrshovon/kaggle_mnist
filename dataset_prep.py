import numpy as np
import csv
import cv2
from tqdm import tqdm
IMGSIZE=28
SZ=IMGSIZE
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def get_one_hot_output(value):
	output=[0,0,0,0,0,0,0,0,0,0]
	output[value]=1
	return output

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
	
def prepare_data(filename,is_labeled=True):
	with open(filename,'r') as f:
		next(f,None)
		reader_obj=csv.reader(f)
		list_rd=list(reader_obj)
		#print len(list_rd)
		#get the labels
		labels=[]
		dataset_list=[]
		im_start_index=0
		if is_labeled==True:
			im_start_index=1
		else:
			im_start_index=0
		for item in tqdm(list_rd):
			#convert it to int
			row=list(map(int,item))
			label=''
			if(is_labeled==True):
				label=row[0]
			img_arr=[]
			for i in range(im_start_index,IMGSIZE*IMGSIZE,IMGSIZE):
				img_arr.append(row[i:i+IMGSIZE])
			#now we have out image
			#append label and image
			img_arr=np.array(img_arr,dtype=np.float32)
			img_arr=deskew(img_arr)
			if is_labeled==True:
				dataset_list.append([img_arr,get_one_hot_output(label)])
			else:
				dataset_list.append([img_arr])
		dataset=np.array(dataset_list)
		np.save(filename+".npy",dataset)
def prepare_train_test_data():
	print("Preparing training data")
	prepare_data("train.csv")
	print("Preparing test data")
	prepare_data("test.csv",is_labeled=False)
	test=np.load('test.csv.npy')
	cv2.imwrite("test.png",test[0][0])
prepare_train_test_data()
