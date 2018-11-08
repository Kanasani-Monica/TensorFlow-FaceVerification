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

from tfmtcnn.networks.FaceDetector import FaceDetector
from tfmtcnn.networks.NetworkFactory import NetworkFactory

from tfface.classifier.Classifier import Classifier

face_detector = None
classifier_object = None

def parse_arguments(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_name', type=str, help='The name of the architecture.', default='inception_v3')    
	parser.add_argument('--checkpoint_path', type=str, help='The directory where the model was written to or an absolute path to a checkpoint file.', default=None)
	parser.add_argument('--dataset_dir', type=str, help='The directory where the dataset files are stored.', default=None)

	parser.add_argument('--image1', type=str, help='image1 ID.', default=None)
	parser.add_argument('--image2', type=str, help='image2 ID.', default=None)
	parser.add_argument('--threshold', type=float, help='Lower threshold value for classification (0 to 1.0).', default=0.12)
	parser.add_argument('--model_root_dir', type=str, help='Input model root directory where model weights are saved.', default=None)

	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
	return(parser.parse_args(argv))

def  features(input_image):
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

	if(args.model_root_dir):
		model_root_dir = args.model_root_dir
	else:
		model_root_dir = NetworkFactory.model_deploy_dir()

	last_network='ONet'
	face_detector = FaceDetector(last_network, model_root_dir)

	classifier_object = Classifier()
	if(not classifier_object.load_dataset(args.dataset_dir)):
		return(False)
	if(not classifier_object.load_model(args.checkpoint_path, args.model_name, args.gpu_memory_fraction)):
		return(False)

	image1 = cv2.imread(args.image1)
	input_image_height, input_image_width, input_image_channels = image1.shape
	#print(input_image_height, input_image_width)
	
	image2 = cv2.imread(args.image2)
	input_image_height, input_image_width, input_image_channels = image2.shape
	#print(input_image_height, input_image_width)
	
	i1 = features(image1)
	i2 = features(image2)
	
	#if(not (i1 and i2)):
		#return(False)
	
	result=l1_loss(i1,i2)
	print ("The answer is " + str(result) )

	result=l2_loss(i1,i2)
	print ("The answer is " + str(result) )

	result=dot_product(i1,i2)
 	print ("The answer is " + str(result) )


if __name__ == '__main__':

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))



