# Object-Detection-
Example code to detect objects via images and your webcam
I learnt about object detection for an internal project at work, however this example is based on this tutorial. 
https://www.murtazahassan.com/courses/opencv-projects/lesson/code-and-files/. 

You can view the full tutorial here:
https://www.youtube.com/watch?v=HXDD7-EnGBY

I have broken down the required steps below:

Step 1: Download required files
coco.names - The model has been pretrained to detect certain objects. The .names file contains all the names of detectable objects. This is the file that will enable your model to provide the relevant name of the object instead of an an object ID such as "object" 1.

frozen_inference_graph.pb - Weights file. The model was trained on a large dataset. During this process, weights for the machine learning model have been tuned and stored within this file. If you wanted to detect new objects, there weights would be changed when you retrain the model. This is generally a large file and you shouldn't be able to read anything when you open it.

All the relevant files can be downloaded from here:
https://www.murtazahassan.com/courses/opencv-projects/lesson/code-and-files/. 

The only module/package required in your virtual environment is opencv.

Create a virtual environment and install the required libary by running the following command in the anaconda cmd.
pip install opencv-python

