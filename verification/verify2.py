# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

r"""Webcamera demo.

Usage:
```shell

$ python webcamera_demo.py \
	--model_name=inception_v3 \
	--checkpoint_path=/tensorflow/models/inception_v3/face_images \
	--dataset_dir=/tensorflow/datasets/face_images 

$ python webcamera_demo.py \
	--model_name=inception_v3 \
	--checkpoint_path=/tensorflow/models/inception_v3/face_images \
	--dataset_dir=/tensorflow/datasets/face_images \
	 --webcamera_id=0 \
	 --threshold=0.125 

$ python webcamera_demo.py \
	--model_name=inception_v3 \
	--checkpoint_path=/tensorflow/models/inception_v3/face_images \
	--dataset_dir=/tensorflow/datasets/face_images \
	 --webcamera_id=0 \
	 --threshold=0.125 \
	 --model_root_dir=/mtcnn/models/mtcnn/deploy/
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
from tfface.classifier.Classifier import Classifier

classifier_object = None

def parse_arguments(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_name', type=str, help='The name of the architecture.', default='inception_v3')    
	parser.add_argument('--checkpoint_path', type=str, help='The directory where the model was written to or an absolute path to a checkpoint file.', default=None)
	parser.add_argument('--dataset_dir', type=str, help='The directory where the dataset files are stored.', default=None)

	parser.add_argument('--directory1', type=str, help='dir1.', default=None)
	parser.add_argument('--directory2', type=str, help='dir2.', default=None)
	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
	return(parser.parse_args(argv))

def features(input_image):
	global classifier_object
	facial_features = classifier_object.features(input_image)
	return(facial_features)
	
def l1_loss(image1,image2):
	geterror=np.absolute(image2-image1)
	totalloss=np.sum(geterror)
	return totalloss

def l2_loss(image1,image2):
	geterror=np.absolute(image2-image1)
	geterror=geterror**2
	totalloss=np.sum(geterror)
	return totalloss

def dot_product(image1,image2):
	#geterror=np.dot(image1,image2)
	image1 = np.array(image1)
	image2 = np.array(image2)
	image2 = np.transpose(image2)
	totalloss= image1.dot(image2)
	return totalloss

def main(args):
	global face_detector
	global classifier_object

	if(not args.checkpoint_path):
		raise ValueError('You must supply the checkpoint path with --checkpoint_path')	
	if(not os.path.exists(args.checkpoint_path)):
		print('The checkpoint path is missing. Error processing the data source without the checkpoint path.')
		return(False)

	if(not args.dataset_dir):
		raise ValueError('You must supply the dataset directory with --dataset_dir')		
	if(not os.path.exists(args.dataset_dir)):
		print('The dataset directory is missing. Error processing the data source without the dataset directory.')
		return(False)

	classifier_object = Classifier()
	if(not classifier_object.load_dataset(args.dataset_dir)):
		return(False)
	if(not classifier_object.load_model(args.checkpoint_path, args.model_name, args.gpu_memory_fraction)):
		return(False)

	directory1 = args.directory1
	onlyfiles1=[f for f in listdir(directory1) if isfile (join(directory1,f))]
	images1=np.empty(len(onlyfiles1),dtype=object)

	for n in range(0,len(onlyfiles1)):
		images1[n] = cv2.imread(join(directory1,onlyfiles1[n]))
	
	directory2 = args.directory2
	onlyfiles2=[f for f in listdir(directory2) if isfile (join(directory2,f))]
	images2=np.empty(len(onlyfiles2),dtype=object)

	for n in range(0,len(onlyfiles2)):
		images2[n] = cv2.imread(join(directory2,onlyfiles2[n]))

	min_l1=1
	max_l1=0
	min_l2=1
	max_l2=0
	min_dot=1
	max_dot=0
	for i in range(0,len(images1)):
		input_image_height, input_image_width, input_image_channels = images1[i].shape
		
		for j in range(0,len(images2)):
			input_image_height, input_image_width, input_image_channels = images2[j].shape

			i1 = features(images1[i])
			i2 = features(images2[j])
	
	                result=l1_loss(i1,i2)
			if(result<min_l1):
				min_l1=result
				#l1_i1=images[i]
				#l1_i2=images[j]
			if(result>max_l1):
				max_l1=result
				#l1_i1=images[i]
				#l1_i2=images[j]
			result=l2_loss(i1,i2)
			if(result<min_l2):
				min_l2=result
			if(result>max_l2):
				max_l2=result	
			result=dot_product(i1,i2)
			if(result > 0.5):
				print('Similar faces are (', result, ') - ', onlyfiles1[i], onlyfiles2[j])
		 	if(result<min_dot):
				min_dot=result
			if(result>max_dot):
				max_dot=result	

	print ("L1 loss minimum value is",min_l1)
	print ("L1 loss maximum value is",max_l1)
	print ("L2 loss minimum value is",min_l2)
	print ("L2 loss maximum value is",max_l2)
	print ("arcosine loss minimum value is",min_dot)
	print ("arcosine loss maximum value is",max_dot)	

if __name__ == '__main__':

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))



