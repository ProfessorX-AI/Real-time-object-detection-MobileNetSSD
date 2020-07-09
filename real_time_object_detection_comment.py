# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments




# initialize the list of class labels MobileNet SSD was trained to detect

            
# generate a set of colors for each class label



# Now we will load our model from disk


# initialize our videostream

# allow the camera to warm up for 2 seconds

# initialize FPS counter




# Now we load our each and evey frame (for speed puposes, you could skip frames)

# loop over the frames from the video stream and

    
    # grab the frame from threaded video stream and resize it
    # to have a maximum width of 400 pixels
    

    # grab the frame dimensions and covert it to a blob 
    

    # pass the blob through the network and obtain the detections and predictions
   

    # Now it's time to look at our confidence values

    

        # extract the confidence (i.e. the probability associated with the prediction)
        

        # filter out weak detections by ensuring the confidence value is greater than the minimum confidence values
        
            # extract the index of the class label from the detections array
            
            # calculate the x y coordinates of the bounding box for the object
            
            

            # draw the prediction on the frame 
            # label to draw on the frame
            
            
            # draw rectangle
            
            
            # calculate the point where to write the predicted class and probability
            
            # write the text on the frame 
            
            
    # show the output frame and
    

    # if the 'q' key was pressed, break from the loop
    
    # update the FPS counter
    

# stop the timer and display FPS imformation




# do a bit of cleanup
