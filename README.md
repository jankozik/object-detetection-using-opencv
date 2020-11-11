# Object-Detection using OpenCV

## Credit to Murtazahassan for his tutorial on Object 
- https://www.murtazahassan.com/courses/opencv-projects/lesson/code-and-files/.
- https://www.youtube.com/watch?v=HXDD7-EnGBY


## Getting Started
<u> Step 1: Download required files </u>
- coco.names - The model has been pretrained to detect certain objects. The .names file contains the names of all detectable objects. Your model will detect an integer and then find the relevant object from the coco.names files

- frozen_inference_graph.pb - Weights file. The model was trained on a large dataset. During this process, weights for the machine learning model have been tuned and stored within this file. If you wanted to detect new objects, there weights would be changed when you retrain the model. This is generally a large file and you shouldn't be able to read anything when you open it.

<u> Step 2: Create the virtual environment
- run the following command relative to your directory to create the environment with the relevant dependencies <br>
conda env create -f environment.yml 

<u> Step 3: Run the main_webcam.py file
python main_webcam.py

