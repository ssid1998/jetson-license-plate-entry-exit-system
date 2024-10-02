# Trainingmodell erstellen

import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense , Dropout , Activation , Flatten
from tensorflow.python.keras.layers import Conv2D , MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import collections # to calculate the class weights
from matplotlib import pyplot
from tqdm import tqdm

import pickle

pickle_in = open("X.pickle" ,"rb")
X = pickle.load( pickle_in )

pickle_in = open("y.pickle" ,"rb")
y = pickle.load( pickle_in )

pickle_in = open("Xval.pickle" ,"rb")
Xval = pickle.load( pickle_in )

pickle_in = open("yval.pickle" ,"rb")
yval = pickle.load( pickle_in )

X = X/255.0
Xval = Xval /255.0

print (X.shape )


IMG_HEIGHT = 80
IMG_WIDTH = 30
datagen = ImageDataGenerator (
            rotation_range = 5 ,
            brightness_range = (0.2 ,1.5) ,
            width_shift_range =0.1 ,
            height_shift_range =0.1 ,
            zoom_range =0.1 ,
            shear_range =0.1 ,
            #horizontal_flip=True,
            fill_mode=’nearest’) # define data prep
datagen.fit(X) # fit statistics parameters from data

print (X [ 0 ].shape )

for X_batch , y_batch in datagen.flow(X , y , batch_size=9) :
    for i in range (0 ,9) :
        pyplot.subplot(330+1+i)
        pyplot.imshow ( X_batch [ i ].reshape ( IMG_HEIGHT , IMG_WIDTH ) , cmap=pyplot.get_cmap (’gray’) )
        pyplot.text (11 , - 5 , str( y_batch [ i ]) )
        pyplot.show ()
    break
    X_batch [ i ].shape


NAME = "demold_16_32_32 -32CNN-aug -50drop-ver180 -single_mold"

model = Sequential ()
model.add( Conv2D (16 , (3 , 3) , input_shape=(IMG_HEIGHT , IMG_WIDTH , 1) , padding=’same’) )
model.add( Activation (’relu’) )
model.add( MaxPooling2D ( pool_size=(2, 2) ) )

model.add( Conv2D (32 , (3 , 3) , padding=’same’) )
model.add( Activation (’relu’) )
model.add( MaxPooling2D ( pool_size=(2, 2) ) )

model.add( Conv2D (32 , (3 , 3) , padding=’same’) )
model.add( Activation (’relu’) )
model.add( MaxPooling2D ( pool_size=(2, 2) ) )

model.add( Flatten () ) # this converts our 3D feature maps to 1D feature vectors
model.add( Dense (32) )
model.add( Activation (’relu’) )
model.add( Dropout (0.5) )

model.add( Dense (1) )
model.add( Activation (’sigmoid’) )

model.compile (loss=’binary_crossentropy’ ,
optimizer=’AdaDelta’ ,
metrics=[’accuracy’ ])

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) # make our TensorBoard
callback object

model.fit_generator ( datagen.flow(X , y , batch_size=64) ,
epochs=40,
steps_per_epoch=200,
validation_data=datagen.flow(Xval , yval , batch_size=64))
#callbacks=[tensorboard])

NAME = "demold_16_32_32 -32-32CNN-aug -50drop-ver180 -single_mold"
model.save(NAME+’.model’)