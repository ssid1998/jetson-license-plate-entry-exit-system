# Programmabschnitt zur Bildauswertung mit Hilfe eins GPIO Signals
# beim NVIDIA Jetson Nano

import cv2
import Jetson.GPIO as GPIO
import time
import numpy as np
import tensorflow as tf
global model

cam1_pin = 18

def get_frame () :

def gstreamer_pipeline (
    capture_width=3280,
    capture_height=2464,
    display_width=640,
    display_height=360,
    framerate=40,
    flip_method=0,
    9 ) :
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width ,
            capture_height ,
            framerate ,
            flip_method ,
            display_width ,
            display_height ,
           )
        )

    print ( gstreamer_pipeline ( flip_method=0))

    cap = cv2.VideoCapture ( gstreamer_pipeline ( flip_method=0) , cv2.CAP_GSTREAMER )

    if cap.isOpened () :
        ret_val , img = cap.read ()
        #cv2.imshow("CSI Camera", img)
        #image_name="opencv_single_frame.png"
        #cv2.imwrite(image_name , img)
        #print("single frame written!")
        #image_counter +=1
        cap.release ()
        #cv2.destroyAllWindows()
    else :
        print ("Die Kamera konnte nicht gefunden werden")
    
	return img

def prepare (img) :
    IMG_HEIGHT = 160
    IMG_WIDTH = 60
    img_array = cv2.cvtColor (img , cv2.COLOR_BGR2GRAY )
    img_array=np.rot90 (img_array , - 1)
    img_array= cv2.resize (img_array , (2∗IMG_WIDTH , IMG_HEIGHT ) )

    imleft=img_array [0: IMG_HEIGHT , 0: IMG_WIDTH ]
    imright=np.fliplr ( img_array [0: IMG_HEIGHT , IMG_WIDTH :2∗ IMG_WIDTH ])
    return imleft.reshape ( - 1 , IMG_HEIGHT , IMG_WIDTH ,1) , imright.reshape ( - 1 , IMG_HEIGHT, IMG_WIDTH ,1)

def main () :
    GPIO.setmode (GPIO.BOARD )
    GPIO.setup (cam1_pin , GPIO.IN)
    GPIO.add_event_detect (cam1_pin , GPIO.RISING , bouncetime=10)
    print ("Starting demo now! Press CTRL+C to exit")

    try :
        model=tf.keras.models.load_model (’ModelEntnahme.model’)
        while True :
            # blink LED 1 slowly
            if GPIO.event_detected ( cam1_pin ) :
                print ("Interrupt for camera triggered on channel {}".format ( cam1_pin ) )
                img = get_frame ()
                imleft , imright = prepare (img)

                CATEGORIES=[’good’ ,’bad’ ]

                prediction_left=model.predict ( imleft )
                prediction_right=model.predict ( imright )

                if CATEGORIES [ int( prediction_left )]==’good’ and CATEGORIES [ int( prediction_right )]==’good’ :
                    print (’Result: left: good , right: good’)
                if CATEGORIES [ int( prediction_left )]==’bad’ and CATEGORIES[ int( prediction_right )]==’good’ :
                    print (’Result: left: bad , right: good’)
                if CATEGORIES [ int( prediction_left )]==’good’ and CATEGORIES [ int( prediction_right )]==’bad’ :
                    print (’Result: left: good , right: bad’)
                if CATEGORIES [ int( prediction_left )]==’bad’ and CATEGORIES[ int( prediction_right )]==’bad’ :
                    print (’Result: left: bad , right: bad’)

    finally :
        GPIO.cleanup ()

if __name__ == ’__main__’ :
    main ()

GPIO.cleanup ()