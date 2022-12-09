# VisualGuidedObjectManipulationWithVREP
Visual guided object manipulation with VREP for IE 57400 project.
## Required Software
- Latest version of Tensorflow
- OpenCV
- Python 3.x.x
- VREP
## Features
- Uses LSTM to predict object's position in VREP simulation to allow end effector to move to object accurately
- Uses VREP Python RemoteApi library to control objects
## Data Collection
Ten images with a colored object captured in VREP is used for the training set under /vrep/Training_Set/training_set_x_x.png
The training set contains image directory and multiple positions (x,y,z) of the object in each image under /vrep/Training_Set/training_set.csv
## Training
The VREP scene "object_manipulation_tasks.ttt" found in /vrep must be opened and simulated prior to training and testing.  The VREP RemoteAPI library allows control over objects within the simulation environment such as robot end effector and camera. 
- simxGetVisionSensorImage to retrieve current camera's frame
- simxCallScriptFunction to call pre-defined scripts (ex. goToPos to have robot arm move end effector to defined position)

Execute "object_manipulation_tasks.py" under /vrep to train LSTM model with training set and test with the current object in the simulation environment. 

![image](https://user-images.githubusercontent.com/26236571/124688923-77b3ec80-de8c-11eb-827e-c10db80e3e10.png)
## Testing
LSTM model is tested by predicting the new object's position and calling "goToPos" to place end effector at the defined position.  Robot arm will grasp and move the object to the drop off position.

![ezgif-6-07adf065c6bf](https://user-images.githubusercontent.com/26236571/124690078-83081780-de8e-11eb-9d98-42ccc6776c18.gif)
# Results

![image](https://user-images.githubusercontent.com/26236571/124690430-0d507b80-de8f-11eb-82d8-d112b984e609.png)
## Improvements
- Current implementation does not utilize LSTM to it's full potential by using VREP to capture the entire range of motion when grabbing and dropping object at designated location.  Lack of knowledge with VREP is a roadblock.  
- Create a more complex simulation environment with wider area of manipulation with multiple elevations.
