#!/usr/bin/python
#
# Script for using tflite-Models with photos from disc
# The tflite-Models were trained with the oroginal CIFAR-10 dataset
# and a manipulated CIFAR-10 dataset (category 'bottle' added from CIFAR-100)
#
# Documentation in the Document 'Kuenstliche Intelligenz mit dem Jetson Nano' by Elmar Wings
# Related to the Project 'Kuenstliche Intelligenz mit dem Jetson Nano und TensorFlow Lite'
#
# Written by C.Joachim in January 2021
# based on the imagenet-console from https://github.com/dusty-nv/jetson-inference
# and TensorFlow Lite inference with Python from https://www.tensorflow.org/lite/guide/inference
# and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# 
# To run enter (in the example using the image bottle_1 and the model1)
# $ python3 ./tflite-foto.py bottle_1.jpg m1_bottle_1.jpg model1.tflite
#
# The Models:
# model1.tflite: manipulated CIFAR-10, AlexNet-architecture, trained in 100 epochs with Batch Size 64, model not quantized
# model2.tflite: manipulated CIFAR-10, AlexNet-architecture, trained in 100 epochs with Batch Size 64, quantized with tflite-converter
# model3.tflite: original CIFAR-10, AlexNet-architecture, trained in 100 epochs with Batch Size 64, model not quantized
# model4.tflite: original CIFAR-10, AlexNet-architecture, trained in 100 epochs with Batch Size 64, quantized with tflite-converter

import jetson.inference
import jetson.utils

import argparse
import sys

import tensorflow as tf
from PIL import Image
import numpy as np
import time

# parse the command line
parser = argparse.ArgumentParser(description="Classify an image using tflite-models with categories airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (and) bottle", 
						   formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
parser.add_argument("model", type=str, default="model1.tflite", help="model to load, in the format <model1.tflite>")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter("tflitemodels/"+opt.model)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Adjust input
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img = Image.open("images/"+opt.file_in).resize((width, height))
input_data = np.expand_dims(img, axis=0)
input_data = (np.float32(input_data) - np.mean(input_data)) / np.std(input_data)

# Use the Model
interpreter.set_tensor(input_details[0]['index'], input_data)
start_time = time.time()
interpreter.invoke()
stop_time = time.time()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Determine probability and index of max. probability
conf=np.amax(output_data)
idx=np.argmax(output_data)

# Find the object description
if opt.model=="model1.tflite" or opt.model=="model2.tflite":
	category=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'bottle']
else:
	category=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_desc = category[idx]

# Overlay the result on the image
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)
if opt.file_out is not None:
	font = jetson.utils.cudaFont(size=jetson.utils.adaptFontSize(width))	
	font.OverlayText(img, width, height, "{:.2f}% {:s} {:.2f}s".format(conf * 100, class_desc,stop_time-start_time), 10, 10, font.White, font.Gray40)
	jetson.utils.cudaDeviceSynchronize()
	jetson.utils.saveImageRGBA("out_images/"+opt.file_out, img, width, height)

# Print out the result
print("\nimage is recognized as '{:s}' (class #{:d}) with {:f}% confidence in {:f}s\n".format(class_desc, idx, conf * 100, stop_time-start_time))
print(idx,output_data)
