# Erstellung der Trainingdaten 

import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
from tqdm import tqdm
from random import sample
import pickle # to save data
import random
import glob
from imblearn.over_sampling import RandomOverSampler # to balance imbalanced datasets
import tensorflow as tf

DATADIR = "./Datenbasis/"
CATEGORIES = [ "good" , "bad" ]
IMG_HEIGHT = 80
IMG_WIDTH = 30

training_data = [ ]
def create_training_data () :
    for category in CATEGORIES :
        path = os.path.join(DATADIR , category )
        class_num = CATEGORIES.index ( category )
            for img in tqdm (os.listdir (path) ) :
                try :
                    img_array = cv2.imread (os.path.join(path , img) ,cv2.IMREAD_GRAYSCALE )
                    new_array = cv2.resize (img_array , (IMG_WIDTH , IMG_HEIGHT ) )
                    training_data.append ([ new_array , class_num ])
                except Exception as e : # in the interest in keeping the output
                    clean...
                    pass

create_training_data ()

print (len( training_data ) )

random.shuffle ( training_data )

Xval = [ ]
yval = [ ]
X = [ ]
y = [ ]
ValNum =300

for features , label in training_data [0: ValNum - 1]:
    Xval.append ( features )
    yval.append ( label )
Xval = np.array (Xval).reshape ( - 1 , IMG_HEIGHT , IMG_WIDTH , 1)

for features , label in training_data [ ValNum : - 1]:
    X.append ( features )
    y.append ( label )
X = np.array(X).reshape ( - 1 , IMG_HEIGHT , IMG_WIDTH , 1)

sm = RandomOverSampler ( random_state=42) # initialize adasyn
X , y = sm.fit_resample (X.reshape ( - 1 , IMG_HEIGHT*IMG_WIDTH ) , y)
X = X.reshape ( - 1 , IMG_HEIGHT , IMG_WIDTH , 1)

pickle_out = open("Xval.pickle" ,"wb")
pickle.dump(Xval , pickle_out )
pickle_out.close ()

pickle_out = open("yval.pickle" ,"wb")
pickle.dump(yval , pickle_out )
pickle_out.close ()

pickle_out = open("X.pickle" ,"wb")
pickle.dump(X , pickle_out )
pickle_out.close ()

pickle_out = open("y.pickle" ,"wb")
pickle.dump(y , pickle_out )
pickle_out.close ()