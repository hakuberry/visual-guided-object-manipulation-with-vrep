import vrep                  
import sys
import time               
import numpy as np       
import math
import matplotlib.pyplot as mpl
import cv2


PI=math.pi  
emptyBuff = bytearray()

vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

if clientID!=-1:  #check if client connection successful
    print('Connected to remote API server')
    
else:
    print('Connection not successful')
    sys.exit('Could not connect')
'''
#retrieve camera  handles
errorCode, cam_handle = vrep.simxGetObjectHandle(clientID, 'cam1', vrep.simx_opmode_oneshot_wait)
    
errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_streaming)

for x in range(1, 10):
    errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_buffer)
    im = np.array(image, dtype=np.uint8)
    print(im.shape)
    im.resize(256, 256, 3)
    im = cv2.flip(im, 0)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
'''

errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToDropPos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
time.sleep(1)
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToHomePos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
time.sleep(1)
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToPos', [], [0.85, -0.8, 0.78], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
time.sleep(1)
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'activateVacuum', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
time.sleep(1)
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToDropPos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
time.sleep(1)
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'deactivateVacuum', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)

#for loop to retrieve sensor arrays and initiate sensors
#for x in range(1,16+1):
#        errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_buffer)
#        im = np.array(image, dtype=np.uint8)
#        print(im.shape)
#        #print(resolution)
#        im.resize(256, 256, 3)
#        im = cv2.flip(im, 0)
#        print(im.shape)
#        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#        cv2.imshow('image', im)

time.sleep(5)        
#close connection
vrep.simxFinish(clientID)
