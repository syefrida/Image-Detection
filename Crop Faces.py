#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from glob import glob


# In[2]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')


# In[3]:


testpath = glob('./data/1mttkb1.jpg')


# In[4]:


len(testpath)


# In[5]:


# load haar cascade classifier
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')


# In[9]:


gambar = cv2.imread('./data/1mttkb1.jpg')

# convert image into grayscale
gray = cv2.cvtColor(gambar,cv2.COLOR_BGR2GRAY)

# load haar cascade classifier
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
faces = haar.detectMultiScale(gray,1.5,3)

jum = 0
for x,y,w,h in faces:
    jum=jum+1
    cv2.rectangle(gambar,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = gray[y:y+h,x:x+w]  #crop bagian wajah dari gambar gray
    
plt.imshow(gambar)
print('Jumlah wajah terdeteksi: ', jum)


# In[ ]:




