# Object-Detection using OpenCV
OpenCV contains methods that can accept object detection weight/config files for a range of different object detection models. OpenCV can output confidence threshold and bounding box coordinates. This is extremely useful as OpenCV has methods that can accept bounding box coordinates to quickly plot boxes around the detected objects and label detected objects with a minimum amount of simple, clean code. OpenCV also has a method to apply Non-maximum Suppression (NMS), a technique used to prevent multiple detections of the same object by only keeping the detection with the highest confidence for that particular object. 

Example detections <br>
![Test](https://github.com/VirajVaitha123/Object-Detection-/blob/master/Images/SampleDetection.PNG)

## Credit to Murtazahassan for his tutorial on Object Detection 
- https://www.murtazahassan.com/courses/opencv-projects/lesson/code-and-files/.
- https://www.youtube.com/watch?v=HXDD7-EnGBY
<br>
<br>
<br>

## Getting Started
<u> Step 1: Download required files </u> <br>
`<addr>` git clone https://github.com/VirajVaitha123/Object-Detection-

- coco.names - The model has been pretrained to detect certain objects. The .names file contains the names of all detectable objects. Your model will detect an integer and then find the relevant object from the coco.names files

- frozen_inference_graph.pb - Weights file. The model was trained on a large dataset. During this process, weights for the machine learning model have been tuned and stored within this file. If you wanted to detect new objects, there weights would be changed when you retrain the model. This is generally a large file and you shouldn't be able to read anything when you open it

- ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt The mobilenet-ssd model is a Single-Shot multibox Detection (SSD) network intended to perform object detection. This file is a config file that contains the relevant parameters for opencv to run the model
<br>
<u> Step 2: Create the virtual environment <br>
- run the following command relative to your directory to create the environment with the relevant dependencies <br>
conda env create -f DataScience.yml 
<br>
<br>
<u> Step 3: Run the main_webcam.py file <br>
python main_webcam.py

