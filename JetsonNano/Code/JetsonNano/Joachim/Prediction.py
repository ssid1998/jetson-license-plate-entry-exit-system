# Vorhersage

import cv2
import os # to deal with paths
import tensorflow as tf
import glob
import numpy
from matplotlib import pyplot
%matplotlib inline
from shutil import move

CATEGORIES = [ ’good’ ,’bad’ ] # will use this to convert prediction num to string value

files = glob.glob(’./testing/bad50x75/*.bmp’) # folder to test for spills

def prepare ( filepath ) :
    IMG_SIZE = 50
    img_array = cv2.imread (filepath , cv2.IMREAD_GRAYSCALE ) # read image , convert to grayscale
    new_array = cv2.resize (img_array , (IMG_SIZE , IMG_SIZE ) ) # resize image to match model’s expected sizing
    return new_array.reshape ( - 1 ,IMG_SIZE , IMG_SIZE ,1) # return image with shaping that TF wants

prediction = numpy.zeros (( len( files ) ,1 , 1) )

for i in range (len( files ) ) :
    prediction [ i ,0 ,0] = model.predict ([ prepare ( files [ i ]) ])
    print ( files [ i]+’: ’+CATEGORIES [ int( prediction [ i ] [ 0 ] [ 0 ] ) ])
    img_array = cv2.imread ( files [ i ] ,cv2.IMREAD_COLOR ) # convert to array
    
pyplot.imshow (img_array , cmap=’gray’) # graph it
pyplot.show () # display!
if CATEGORIES [ int( prediction [ i ] [ 0 ] [ 0 ] ) ] == ’bad’ :
    pyplot.imshow (img_array , cmap=’gray’) # graph it
    pyplot.show () # display!
     human_input = input ("Is this a spill? (y,n) ")
if human_input == ’y’ :
    move( files [ i ] , ’./bad/’+os.path.basename ( files [ i ]) )