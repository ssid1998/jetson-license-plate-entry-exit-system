# MIT License 
# Copyright (c) 2019 JetsonHacks 
# See license 
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a 
# NVIDIA Jetson Nano Developer Kit using OpenCV 
# Drivers for the camera and OpenCV are included in the base image 

# The program continuously captures frames from the videostream
# Importing CV2 module of OpenCV
import cv2 
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera 
# Defaults to 1280x720 @ 60fps 
# Flip the image by setting the flip_method (most common values: 0 and 2) 
# display_width and display_height determine the size of the window on the screen 
def gstreamer_pipeline( 
    capture_width=1280, 
    capture_height=720, 
    display_width=1280, 
    display_height=720, 
    framerate=1, 
    flip_method=2, 
): 

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
            capture_width, 
            capture_height, 
            framerate, 
            flip_method, 
            display_width, 
            display_height, 
        ) 

    ) 
def show_camera(): 
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common) 
    count = 0 
    print(gstreamer_pipeline(flip_method=2)) 
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER) 
    if cap.isOpened(): 
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE) 
        # Window 
        while cv2.getWindowProperty("CSI Camera", 0) >= 0: 
            ret_val, img = cap.read() 
            #ret_val is a boolean variable that returns true if the frame is available.
            #frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
            cv2.imshow("CSI Camera", img)           
            #The frames captured are stored in a sequence frame1, frame2, frame3...
            cv2.imwrite("/home/msrjetson/Desktop/Akhila/capture/frame%d.jpg" % count , img) 
            count += 1 
            # This also acts as 
            keyCode = cv2.waitKey(30) & 0xFF 
            # Stop the program on the ESC key 
            if keyCode == 27: 
                break 
        cap.release() 
        cv2.destroyAllWindows() 
    else: 
        print("Unable to open camera")    
if __name__ == "__main__": 

    show_camera() 