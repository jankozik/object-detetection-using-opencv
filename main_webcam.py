#Import relevant modules
import cv2

#Read in the image file
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)



#Import the class names
classNames = []
classFile = 'config_files/coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#Import the config and weights file
configPath = 'config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #Important parameters for the model
weightsPath = 'config_files/frozen_inference_graph.pb'                   #Weights derived from training on large objects dataset

#Set relevant parameters
net = cv2.dnn_DetectionModel(weightsPath,configPath)
#These are some suggested settings from the tutorial, others are fine but this can be used as a baseline
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, image = cap.read()
    classIds, confs, bbox = net.detect(image, confThreshold = 0.5)
    print(classIds, confs,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(image,box,color=(0,255,0), thickness=2)
            cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    #Open Video
    cv2.imshow("Output", image)
    cv2.waitKey(1)