#!/usr/bin/python
#



import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using tflite-models with categories airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (and) bottle", 
formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
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


# capture the image
img, width, height = camera.CaptureRGBA(zeroCopy=1)
jetson.utils.cudaDeviceSynchronize()
display.RenderOnce(img, width, height)

#save the image		
jetson.utils.saveImageRGBA("fotos/"+opt.file_out,img,width,height)

