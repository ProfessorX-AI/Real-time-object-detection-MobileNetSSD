# Real-time-object-detection-MobileNetSSD

## Popular deep learning based object detection models are
- Faster R-CNNs
- YOLO(You Only Look Once)
- Single Shot Detectors

### Faster R-CNNs
- Highly accurate 
- Even with the ‘Faster’ implementation quite slow.
- Speed is about  7 FPS
- Takes at list 100ms per image
- Can be used when speed is not concern.
- Difficult to understand especially for beginners.
[learn more about Faster R-CNNs](https://arxiv.org/abs/1506.01497)

### YOLO
Much faster than the Faster R-CNNs
Capable of processing 40- 90 FPS.
Superfast variant can give 155 FPS.
Accuracy is compromised.
Can be used for purely speed.
[learn more about YOLO](https://arxiv.org/abs/1506.02640)

### Single Shot Detectors(SSD)
Developed by google.
Good balance between the Faster R-CNNs and YOLO.
Algorithm is straightforward than Faster R-CNNs.
Much faster throughput than Faster R-CNNs.
Speed is 22-46 FPS depends on the different variant used.
More accurate than YOLO.
[learn more about SSD](https://arxiv.org/abs/1512.02325)

#### The problem is that these models are very large in the ordre of 200-500 MB. Hence cannot be used for resource constraint devices such as raspberry-pi and Smartphones. The solution is MobileNets.

### MobileNets
Another paper by researchers at google.
Developed for resource constraint devices such as raspberry-pi and smartphones.
Hence the MobileNets.
Have to compromise the accuracy.
[learn more about MobileNets](https://arxiv.org/abs/1704.04861)

### MobileNet SSD
Combination gives the best trade off within the fastest detectors.
Gives the fast and efficient deep learning best object detection method.
First trained on [COCO (common objects in context)](http://cocodataset.org/) dataset.
Fine tuned on PASCAL VOC dataset.
Can detect only 20 objects in images(+1 for background)
Classes are airplanes, bicycles, birds, boats, bottles, buses, cars, cats, chairs, cows, dining tables, dogs, horses, motorbikes, people, potted plants, sheep, sofas, trains, and tv monitors.
[learn more about original tensorflow implementation of MobileNet SSD](https://github.com/Zehaos/MobileNet)

## Let's get to the actual coding
##### requirements

`pip install opencv-python`
`pip install opencv-contrib-python`
`pip install imutils`
`pip install numpy`






