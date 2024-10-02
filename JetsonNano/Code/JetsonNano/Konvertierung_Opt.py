#!/usr/bin/env python
# coding: utf-8

# Script for converting a trained model (see Script Training_CIFAR.py)  
# to a TensorFlow-Lite-Model with optimization  
# 
# Based on https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter

# In[ ]:


import tensorflow as tf


# In[ ]:


#Choose parameters of the trained model to be converted (see script Training_CIFAR)
Dataset = 2
Model   = 2
Epochs  = 100
Batch   = 32


# In[ ]:


#Convert the model
converter               = tf.lite.TFLiteConverter.from_saved_model("saved_model/"+"Dataset %s Model %s Epochs %s Batch Size %s" %(Dataset, Model, Epochs, Batch))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model            = converter.convert()


# In[ ]:


#Save the model.
with open('model2.tflite', 'wb') as f: #choose filename
  f.write(tflite_model)


# In[ ]:




