# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--prototxt", required=True, help="path to caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to pre trained model")
ap.add_argument("-c", "--confidence", type= float, default=0.4, 
                    help="minimum probability to filter weak detections")
                    
args = vars(ap.parse_args())


# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
            'bottle', 'bus', 'car','cat', 'chair', 'cow', 'dinningtable',
            'dog', 'horse', 'motorbike', 'person','pottedplant','sheep', 
            'sofa','train','tvmonitor']
            
# generate a set of colors for each class label

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Now we will load our model from disk
print("[INFO] Loading model from disk...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# initialize our videostream
print('[INFO] starting web camera')
vs = VideoStream(src = 0).start()
# allow the camera to warm up for 2 seconds
time.sleep(2.0)
# initialize FPS counter
fps = FPS().start()



# Now we load our each and evey frame (for speed puposes, you could skip frames)

# loop over the frames from the video stream and
while True:
    
    # grab the frame from threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width = 600)

    # grab the frame dimensions and covert it to a blob 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300,300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections  = net.forward()

    # Now it's time to look at our confidence values

    for i in np.arange(0, detections.shape[2]):

        # extract the confidence (i.e. the probability associated with the prediction)
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence value is greater than the minimum confidence values
        if confidence > args['confidence']:
            # extract the index of the class label from the detections array
            idx = int(detections[0, 0, i, 1])
            # calculate the x y coordinates of the bounding box for the object
            
            box = detections[0, 0,i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) =box.astype(int)

            # draw the prediction on the frame 
            # label to draw on the frame
            label ='{}: {:.2f}%'.format(CLASSES[idx], confidence* 100)
            
            # draw rectangle
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                            COLORS[idx],2)
            
            # calculate the point where to write the predicted class and probability
            y = startY- 15 if startY- 15 > 15 else startY + 15
            # write the text on the frame 
            cv2.putText(frame, label, (startX, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
    # show the output frame and
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break
    # update the FPS counter
    fps.update()

# stop the timer and display FPS imformation
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()