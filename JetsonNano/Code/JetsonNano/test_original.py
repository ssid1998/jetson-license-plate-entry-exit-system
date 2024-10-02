#!/usr/bin/env python
# coding: utf-8

# Script fot testing the trained models with AlexNet Architecture on a chosen picture  
# Written by C.Joachim January 2021  

# In[ ]:


import tensorflow as tf
import numpy as np
import time
from PIL import Image


# In[ ]:


# choose model parameters
Dataset = 2
Model   = 2 # must not be changed
Epochs  = 100
Batch   = 32


# In[ ]:


# load trained model
model = tf.keras.models.load_model("saved_model/"+"Dataset %s Model %s Epochs %s Batch Size %s" %(Dataset, Model, Epochs, Batch))


# In[ ]:


# choose image
file_in = "bottle3.jpg"


# In[ ]:


# prepare imgage - adjust size and normalize
img = tf.io.read_file("images/"+file_in)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.reshape(img,(1,img.shape[0],img.shape[1],3))
img = tf.image.per_image_standardization(img)
img = tf.image.resize(img, (227,227))


# In[ ]:


# evaluate image with model
prob = model.predict(img)


# In[ ]:


# find class with maximum confidence
conf  = np.amax(prob)
index = np.argmax(prob)


# In[ ]:


print(prob)
print(conf)
print(index)


# In[ ]:


# find according class description
if Dataset==2:
    category = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'bottle']
else:
    category = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[ ]:


class_desc = category[index]
print(class_desc)


# In[ ]:




