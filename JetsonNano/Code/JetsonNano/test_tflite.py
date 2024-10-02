#!/usr/bin/env python
# coding: utf-8

# Script for test using the trained models as .tflite with chosen image  
# Written by C.Joachim January 2021  
# 
# Based on https://www.tensorflow.org/lite/guide/inference  
# (section Load and run a model in Python)

# In[ ]:


import tensorflow as tf
import numpy as np
import time
from PIL import Image


# In[ ]:


# chose model and image
model   = "model3.tflite"
file_in = "bottle0.jpg"


# In[ ]:


# initialize tflite-interpreter and get details
interpreter    = tf.lite.Interpreter("tflitemodels/"+model)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[ ]:


# prepare image - adjust size and normalize
# prepare imgage - adjust size and normalize
img = tf.io.read_file("images/"+file_in)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.reshape(img,(1,img.shape[0],img.shape[1],3))
img = tf.image.per_image_standardization(img)
input_data = tf.image.resize(img, (227,227))


# In[ ]:


# evaluate image with model and stop time
interpreter.set_tensor(input_details[0]['index'], input_data)
start_time  = time.time()
interpreter.invoke()
stop_time   = time.time()
output_data = interpreter.get_tensor(output_details[0]['index'])


# In[ ]:


# find class with maximum confidence
conf  = np.amax(output_data)
index = np.argmax(output_data)


# In[ ]:


print(output_data)
print(conf)
print(index)


# In[ ]:


# find according class description
if model=="model1.tflite" or model=="model2.tflite":
    category=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'bottle']
else:
    category=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[ ]:


class_desc = category[index]
print(class_desc)


# In[ ]:




