#Import relevant modules
import cv2
import numpy as np

#Read in the image file
thres = 0.45 #Threshold to detect object
nms_threshold = 0.5 #NMS
cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)
cap.set(10,150)


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
    classIds, confs, bbox = net.detect(image,confThreshold = thres)
    bbox = list(bbox) #NMS function required bbox as a list, not a tuple
    confs = list(np.array(confs).reshape(1,-1)[0]) #[0] removed extra bracket, and reshape used to get the values on the same row
    confs = list(map(float,confs))
    print(classIds, confs,bbox)

    indicies = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    for i in indicies:
        i = i[0] #Get the bounding box info
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(image,(x,y),(x+w,h+y),color = (0,255,0), thickness =2)
        cv2.putText(image,classNames[classIds[i][0]-1],(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
 
    #Without NMS
    #if len(classIds) != 0:
        #for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            #cv2.rectangle(image,box,color=(0,255,0), thickness=2)
            #cv2.putText(image,classNames[classId-1],(box[0]+10,box[1]+30),
                        #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


    