#!/usr/bin/python
#
# Script for using tflite-Models with connected camera in 'live'-mode
# Only classify the image whenever key 'q' is pressed (then take a photo and analyze)
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
# Need to install Keyboard Module with
# sudo pip3 install keyboard
#
# Can be run with 
# $ sudo python3 ./tflite-camera-button.py model1.tflite
# where the latter defines the model to use. 
# For further options see parsing of command line
#
# Create a folder named 'out_images_camera' on same level beforehands
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
import keyboard


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

# Set Pin for Detection Running to High
GPIO.output(Running, GPIO.HIGH)

# process frames until user exits
a=0
first=1
while display.IsOpen():

	# capture the image
	img, width, height = camera.CaptureRGBA(zeroCopy=1)
	frame=jetson.utils.cudaToNumpy(img, width, height, 4)

	# update the title bar
	display.SetTitle("{:s}".format(opt.model))
	
	# counter for rendering image
	a=a+1
	
	# overlay text
	if a>100:
		font.OverlayText(img, width, height, "Press 'q' to take a photo and classify image".format(), 5, 5, font.White, font.Gray40)
	# if first detection has already taken place
	elif a>0 and first==0:
		# if bottle detected
		if idx == 10:
			font.OverlayText(img, width, height, "{:05.2f}% {:s} \n bottle detected".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
		# if no bottle detcted
		else:    
			font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100,     class_desc), 5, 5, font.White, font.Gray40)
		
	# Render image
	display.RenderOnce(img, width, height)
		
	# if 'q' is pressed, take a photo and classify it
	# overlay result and save photo with result
	if keyboard.is_pressed('q'):
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
		confidence=np.amax(output_data)
		idx=np.argmax(output_data)

		# Find the object description
		category=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'bottle']
		class_desc = category[idx]

		# for test purposes, can also be commented out
		print(confidence, idx)
		
		# overlay result on image
		# if bottle is detected
		if idx == 10:
			font.OverlayText(img, width, height, "{:05.2f}% {:s} \n bottle detected".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
			GPIO.output(BottleDet, GPIO.HIGH)
		# if no bottle detcted
		else:    
			font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100,     class_desc), 5, 5, font.White, font.Gray40)
			GPIO.output(BottleDet, GPIO.LOW)
		jetson.utils.cudaDeviceSynchronize()
		
		# save the shot
		t = time.time()
		t = time.strftime("%b-%d-%Y-%H:%M:%S", time.gmtime(t))
		jetson.utils.saveImageRGBA("out_images_camera/"+t+".jpg", img, width, height)
		
		# render image
		display.RenderOnce(img, width, height)

		# reset counter
		a=0

		# first detection completed
		first=0

	
# when the program is exit, reset all GPIOs
atexit.register(GPIO.cleanup)
atexit.register(GPIO.output,Running, GPIO.LOW)
atexit.register(GPIO.output,BottleDet, GPIO.LOW)



