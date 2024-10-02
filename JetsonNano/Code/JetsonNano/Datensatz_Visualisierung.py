#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[2]:


(X_train10, y_train10), (X_test10, y_test10)     = tf.keras.datasets.cifar10.load_data()
(X_train100, y_train100), (X_test100, y_test100) = tf.keras.datasets.cifar100.load_data()


# In[3]:


print('Images Shape: {}'.format(X_train10.shape))
print('Labels Shape: {}'.format(y_train10.shape))
print('Images Shape: {}'.format(X_train100.shape))
print('Labels Shape: {}'.format(y_train100.shape))


# In[4]:


print(y_train10[:10])


# In[5]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train10[i], cmap=plt.cm.binary)
    plt.xlabel(y_train10[i])
plt.show()


# In[6]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train100[i], cmap=plt.cm.binary)
    plt.xlabel(y_train100[i])
plt.show()


# In[7]:


idx        = (y_train100 == 9).reshape(X_train100.shape[0])
X_train100 = X_train100[idx]
y_train100 = y_train100[idx]
for i in range(len(y_train100)):
    y_train100[i]=10


# In[8]:


len(X_train100)


# In[9]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train100[i], cmap=plt.cm.binary)
    plt.xlabel(y_train100[i])
plt.show()


# In[10]:


X_train10_red = [None]*5000
y_train10_red = [None]*5000

for i in range(10):
    idx = (y_train10 == i).reshape(X_train10.shape[0])
    x   = X_train10[idx]
    y   = y_train10[idx]
    X_train10_red[i*500:i*500+500] = x[0:500]
    y_train10_red[i*500:i*500+500] = y[0:500]


# In[11]:


X_train = np.concatenate((X_train10_red, X_train100))
y_train = np.concatenate((y_train10_red, y_train100))


# In[12]:


len(X_train)


# In[13]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i+4998], cmap=plt.cm.binary)
    plt.xlabel(y_train[i+4998])
plt.show()


# In[14]:


shuffler = np.random.permutation(len(X_train))
X_train  = X_train[shuffler]
y_train  = y_train[shuffler]


# In[15]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i+4998], cmap=plt.cm.binary)
    plt.xlabel(y_train[i+4998])
plt.show()


# In[16]:


X_train[1,1:3,1:3,1]


# In[17]:


X_train[1,0:3,0:3,1]


# In[18]:


print(X_train.shape)
print(y_train.shape)
print(y_train[1])
print(X_train[1].shape)


# In[ ]:




