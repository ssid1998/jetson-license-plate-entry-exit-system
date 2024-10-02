#!/usr/bin/env python
# coding: utf-8

# Script for converting a trained model (see Script Training_CIFAR.py)  
# to a TensorFlow-Lite-Model without optimization  
# 
# Based on https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter

# In[2]:


import tensorflow as tf


# In[1]:


#Choose parameters of the trained model to be converted (see script Training_CIFAR)
Dataset = 2
Model   = 2
Epochs  = 100
Batch   = 32


# In[3]:


#Convert the model
converter    = tf.lite.TFLiteConverter.from_saved_model("saved_model/"+"Dataset %s Model %s Epochs %s Batch Size %s" %(Dataset, Model, Epochs, Batch))
tflite_model = converter.convert()


# In[4]:


#Save the model.
with open('model1.tflite', 'wb') as f:
  f.write(tflite_model)


# In[ ]:




