#!/usr/bin/python
#
# Script for using tflite-Models with connected camera in 'live'-mode,
# 50 frames are classified and the category with the highest possibility is displayed as a result
# If a bottle is detected, the chosen Pin is set to high
# Another Pin indicates whether program is ready to execute detection
# Pins to be used as output can be defined below importing commands
#
# The tflite-Models were trained with a manipulated CIFAR-10 dataset (category 'bottle' added from CIFAR-100)
#
# Documentation in the Document 'Kuenstliche Intelligenz mit dem Jetson Nano' by Elmar Wings
# Related to the Project 'Kuenstliche Intelligenz mit dem Jetson Nano und TensorFlow Lite'
# 
# Written by C. Joachim in January 2021
# Based on the modified imagenet-camera for bottle-detection from previous project (bottle.py)
# and TensorFlow Lite inference with Python from https://www.tensorflow.org/lite/guide/inference
# and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# original imagenet-camera from https://github.com/dusty-nv/jetson-inference
#
# The Models:
# model1.tflite: manipulated CIFAR-10, AlexNet-architecture, trained in 100 epochs with Batch Size 64, model not quantized
# model2.tflite: manipulated CIFAR-10, AlexNet-architecture, trained in 100 epochs with Batch Size 64, quantized with tflite-converter
#
# Can be run with 
# $ python3 ./tflite-camera.py model1.tflite
# where the latter defines the model to use. 
# For further options see parsing of command line
#
#

import jetson.inference
import jetson.utils

import argparse
import sys

import atexit
import Jetson.GPIO as GPIO

import numpy as np
import time


BottleDet = 12
Running = 13

#prepare the Pins as Outputs
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BottleDet, GPIO.OUT, initial=GPIO.LOW) #Signal for bottle detected
GPIO.setup(Running, GPIO.OUT, initial=GPIO.LOW) #Signal for image detection running

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using tflite-models with categories airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (and) bottle", 
formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("model", type=str, default="model1.tflite", help="model to load, in the format <model1.tflite>")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
	
# create the camera and display
font = jetson.utils.cudaFont()
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()
 
import tensorflow

# Load the TFLite model and allocate tensors.
interpreter = tensorflow.lite.Interpreter("tflitemodels/"+opt.model)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input details
height_m = input_details[0]['shape'][1]
width_m = input_details[0]['shape'][2]

# Create empty list for index and confidence
class_idx_l = [None]*50
confidence_l = [None]*50

# Set Pin 13 to High
GPIO.output(13, GPIO.HIGH)

# process frames until user exits
a = 0
while display.IsOpen():
	t = 0
	while t<50:
		# capture the image
		img, width, height = camera.CaptureRGBA(zeroCopy=1)
		frame=jetson.utils.cudaToNumpy(img, width, height, 4)
		
		# Adjust input for model
		frame = np.resize(frame,(width_m, height_m,3))
		input_data = np.expand_dims(frame, axis=0)
		input_data = (np.float32(input_data) - np.mean(input_data)) / np.std(input_data)

		# Use the Model
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		# Determine probability and index of max. probability
		conf=np.amax(output_data)
		idx=np.argmax(output_data)

		class_idx_l[t] = idx
		confidence_l[t] = conf
		
		# for test purposes, can also be commented out
		print(conf, idx)
		
		t = t+1

		# render the image and overlay the result from previous run on the image (not in first run)
		if a == 1:
			# if bottle is detected
			if class_idx == 10:
				font.OverlayText(img, width, height, "{:05.2f}% {:s} \n bottle detected".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
			# if no bottle detcted
			else:    
				font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100,     class_desc), 5, 5, font.White, font.Gray40)
			display.RenderOnce(img, width, height)

	# use class index with highest confidence
	confidence = max(confidence_l)
	class_idx = class_idx_l[confidence_l.index(max(confidence_l))]
	

	# Find the object description
	category=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'bottle']
	class_desc = category[idx]

	# render the image and update Pin
	# if bottle detected
	if class_idx == 10:
		font.OverlayText(img, width, height, "{:05.2f}% {:s} \n bottle detected".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
		GPIO.output(BottleDet, GPIO.HIGH)
	# if no bottle detcted
	else:    
		font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100,     class_desc), 5, 5, font.White, font.Gray40)
		GPIO.output(BottleDet, GPIO.LOW)
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s}".format(opt.model))

	# first run finished
	a = 1
	
# when the program is exit, reset all GPIOs
atexit.register(GPIO.cleanup)
atexit.register(GPIO.output,Running, GPIO.LOW)
atexit.register(GPIO.output,BottleDet, GPIO.LOW)



