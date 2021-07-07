import tensorflow as tf
import numpy as np
import csv
import os
import cv2
import time
import math
import sys
import vrep

tf.set_random_seed(777)
emptyBuff = bytearray()

vrep.simxFinish(-1) # close all opened connections
clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

#Training set directory
training_dir = "./Training_Set/training_set.csv"

if clientID != -1:
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

#retrieve parent robot handle
errorCode, main_handle = vrep.simxGetObjectHandle(clientID, 'Baxter_leftArm_joint1', vrep.simx_opmode_oneshot_wait)

#retrieve vacuum handle
errorCode, vac_handle = vrep.simxGetObjectHandle(clientID, 'BaxterVacuumCup#0', vrep.simx_opmode_oneshot_wait)

#retrieve camera  handles
errorCode, cam_handle = vrep.simxGetObjectHandle(clientID, 'cam1', vrep.simx_opmode_oneshot_wait)   
errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_streaming)

def GetObjectCenter(image_dir):
    img = cv2.imread(image_dir, 1)
    center = []
    
    #Lower and upper bound for color (B,G,R)
    lower = [17, 90, 5]
    upper = [80, 255, 60]
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset = (0, 0))

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        m = cv2.moments(c)
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        center.append(cX)
        center.append(cY)
        return center

#Training parameters
seq_length = 1
data_dim = 5
hidden_dim = 5
output_dim = 3
learning_rate = 0.1
iterations = 8000

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
graph = tf.Graph()

#Build the dataset
dataObjectX = []
dataObjectY = []
dataPosX = []
dataPosY = []
dataX = []
dataY = []
dataTemp = []
dataNew1 = []
dataNew2 = []

dataObject = np.loadtxt(training_dir, dtype = 'str', delimiter = ',', skiprows = 1, usecols = [0])
#print(dataObject)
dataPos = np.loadtxt(training_dir, delimiter = ',', skiprows = 1, usecols = range(1, (output_dim + 1)))
#print(dataAngles)


for i in range(0, len(dataObject) - seq_length):
    temp = []
    
    for j in range(i, i + seq_length):
        centerTemp = GetObjectCenter(dataObject[i])
        temp.append(centerTemp)
                    
    _x = temp
    _y = dataObject[i + seq_length] #Next closest image to previous images
    centerY = GetObjectCenter(_y)
    dataObjectX.append(temp)
    dataObjectY.append(centerY)

for i in range(0, len(dataObject)):
    temp1 = []
    temp2 = []
    centerTemp = GetObjectCenter(dataObject[i])
    temp1.append(centerTemp[0] * 0.001)
    temp1.append(centerTemp[1] * 0.001)
    temp1.append(dataPos[i][0])
    temp1.append(dataPos[i][1])
    temp1.append(dataPos[i][2])
    temp2.append(dataPos[i][0])
    temp2.append(dataPos[i][1])
    temp2.append(dataPos[i][2])
    dataNew1.append(temp1)
    dataNew2.append(temp2)

for i in range(0, len(dataPos) - seq_length):
    _x = dataNew1[i:i + seq_length]
    _y = dataNew2[i + seq_length] #Next closest angles 
    dataPosX.append(_x)
    dataPosY.append(_y)

#Train/test split
train_size = int(len(dataPosY) * 0.95)
test_size = len(dataPosY) - train_size
trainX, testX = np.array(dataPosX[0:train_size]), np.array(
    dataPosX[train_size:len(dataPosX)])
trainY, testY = np.array(dataPosY[0:train_size]), np.array(
    dataPosY[train_size:len(dataPosY)])

#Input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

#Build LSTM Network
#Minimize amount of hidden layers because of training time 
cell = tf.nn.rnn_cell.LSTMCell(
        num_units = hidden_dim, state_is_tuple = True, activation = tf.nn.softmax)
multi_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype = tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:,-1], output_dim, activation_fn = None)

#Cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#RMSE
targets = tf.placeholder(tf.float32, [None, output_dim])
predictions = tf.placeholder(tf.float32, [None, output_dim])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #Training
    print('Training...')
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict = {
            X: trainX, Y: trainY})
        print("[Step: {}] Loss: {}".format(i, step_loss))
        test_predict = sess.run(Y_pred, feed_dict = {X: testX})
        rmse_val = sess.run(rmse, feed_dict = {
        targets: testY, predictions: test_predict})
        errX = (abs((testY[0][0] - test_predict[0][0])/test_predict[0][0]) * 100)
        errY = (abs((testY[0][1] - test_predict[0][1])/test_predict[0][1]) * 100)
        errZ = (abs((testY[0][2] - test_predict[0][2])/test_predict[0][2]) * 100)

    #retrieve image from cam1
    print('Retrieving image from cam1...')
    for x in range(1, 5):
        errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        im.resize(256, 256, 3)
        im = cv2.flip(im, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_filename = './current_frame.png'
    cv2.imwrite(im_filename, im)
    
    #predict position
    temp = []
    test1 = []
    test2 = []
    pos = GetObjectCenter(im_filename)
    temp.append(pos[0] * 0.001)
    temp.append(pos[1] * 0.001)
    temp.append(0.875)
    temp.append(-0.800)
    temp.append(0.775)
    test1.append(temp)

    _t = test1[0:1]
    test2.append(_t)
    testX = np.array(test2[0:1])
    
    test_predict = sess.run(Y_pred, feed_dict = {X: testX})
    rmse_val = sess.run(rmse, feed_dict = {
        targets: testY, predictions: test_predict})
    print(test_predict)

    errorCode = vrep.simxSetObjectPosition(clientID, main_handle, vac_handle, {test_predict[0][0], test_predict[0][1], test_predict[0][2]+0.4}, vrep.simx_opmode_oneshot)
    time.sleep(1)
    errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToPickPos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)

'''
#restore trained model
with tf.Session(config=config, graph = graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph('./trained_model-8000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('Placeholder:0')
    y = graph.get_tensor_by_name('Placeholder_1:0')

    #retrieve image from cam1
    for x in range(1, 5):
        errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_buffer)
        im = np.array(image, dtype=np.uint8)
        im.resize(256, 256, 3)
        im = cv2.flip(im, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_filename = 'current_frame.png'
    cv2.imwrite(im_filename, im)

    #predict position from image
    temp = []
    test1 = []
    test2 = []
    pos = GetObjectCenter(im_filename)
    temp.append(pos[0] * 0.001)
    temp.append(pos[1] * 0.001)
    temp.append(0.875)
    temp.append(-0.800)
    temp.append(0.775)
    test1.append(temp)

    _t = test1[0:1]
    test2.append(_t)
    testX = np.array(test2[0:1])
    print(testX)
    
    test_predict = sess.run(Y_pred, feed_dict = {X: testX})
    print(test_predict)
    errorCode = vrep.simxSetObjectPosition(clientID, main_handle, vac_handle, {test_predict[0][0], test_predict[0][1], test_predict[0][2]+0.4}, vrep.simx_opmode_oneshot)
    time.sleep(1)
    errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToPickPos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
'''    
        
