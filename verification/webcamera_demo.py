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

def parse_arguments(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_name', type=str, help='The name of the architecture.', default='inception_v3')    
	parser.add_argument('--checkpoint_path', type=str, help='The directory where the model was written to or an absolute path to a checkpoint file.', default=None)
	parser.add_argument('--dataset_dir', type=str, help='The directory where the dataset files are stored.', default=None)

	parser.add_argument('--webcamera_id', type=int, help='Webcamera ID.', default=0)
	parser.add_argument('--threshold', type=float, help='Lower threshold value for classification (0 to 1.0).', default=0.12)
	parser.add_argument('--model_root_dir', type=str, help='Input model root directory where model weights are saved.', default=None)

	parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
	return(parser.parse_args(argv))

def main(args):

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

	webcamera = cv2.VideoCapture(args.webcamera_id)
	webcamera.set(3, 600)
	webcamera.set(4, 800)
	
	image = cv2.imread('/git-space/16.jpg')
	input_image_height, input_image_width, input_image_channels = image.shape
	print(input_image_height, input_image_width)
		
	face_probability = 0.75
	minimum_face_size = 24
	while True:
    		start_time = cv2.getTickCount()
    		status, current_frame = webcamera.read()

		is_busy = False
    		if status:
        		current_image = np.array(current_frame)
			image_clone = np.copy(current_image)

			if(is_busy):
				continue

			is_busy = True
        		boxes_c, landmarks = face_detector.detect(image_clone)

			end_time = cv2.getTickCount()
        		time_duration = (end_time - start_time) / cv2.getTickFrequency()
        		frames_per_sec = 1.0 / time_duration
        		cv2.putText(current_frame, '{:.2f} FPS'.format(frames_per_sec), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        		for index in range(boxes_c.shape[0]):
            			bounding_box = boxes_c[index, :4]
            			probability = boxes_c[index, 4]            			           
				
				crop_box = []
            			if( probability > face_probability ):
					height, width, channels = image_clone.shape

					crop_box = [int(max(bounding_box[0],0)), int(max(bounding_box[1],0)), int(min(bounding_box[2], width)), int(min(bounding_box[3], height))]
					cropped_image = image_clone[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:]

					crop_height, crop_width, crop_channels = cropped_image.shape
					if(crop_height < minimum_face_size) or (crop_width < minimum_face_size):
						continue

					cv2.rectangle(image_clone, (crop_box[0], crop_box[1]),(crop_box[2], crop_box[3]), (0, 255, 0), 1)

					class_names_probabilities = classifier_object.classify(cropped_image, 1)
					predicted_name = class_names_probabilities[0][0]
					probability = class_names_probabilities[0][1]
					
					if(probability > args.threshold):
						cv2.putText(image_clone, predicted_name + ' - {:.2f}'.format(probability), (crop_box[0], crop_box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
		        cv2.imshow("", image_clone)
			is_busy = False

        		if cv2.waitKey(1) & 0xFF == ord('q'):
            			break
    		else:
        		print('Error detecting the webcamera.')
        		break

	webcamera.release()
	#cv2.destroyAllWindows()


if __name__ == '__main__':

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
	main(parse_arguments(sys.argv[1:]))

