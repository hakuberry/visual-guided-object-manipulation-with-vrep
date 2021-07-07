import tensorflow as tf
import numpy as np
import csv
import os
import cv2
import time
import random
import matplotlib
import matplotlib.pyplot as plt
import vrep
import sys

tf.set_random_seed(777)

#Training set directory
training_dir = "./Training_Set/training_set.csv"

vrep.simxFinish(-1) # close all opened connections
clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
emptyBuff = bytearray()

#retrieve parent robot handle
errorCode, main_handle = vrep.simxGetObjectHandle(clientID, 'Baxter_leftArm_joint1', vrep.simx_opmode_oneshot_wait)

#retrieve vacuum handle
errorCode, vac_handle = vrep.simxGetObjectHandle(clientID, 'BaxterVacuumCup#0', vrep.simx_opmode_oneshot_wait)

#retrieve camera  handles
errorCode, cam_handle = vrep.simxGetObjectHandle(clientID, 'cam1', vrep.simx_opmode_oneshot_wait)   
errorCode, resolution, image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_streaming)

#initialize to home position
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToDropPos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
time.sleep(5)
errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToHomePos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)

def MinMaxNormalize(data):
    num = data - np.min(data, 0)
    den = np.max(data, 0) - np.min(data, 0)
    return num / (den + 1e-7)

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
        #Draw contours when cv2.imshow() is available
        #cv2.drawContours(output, contours, -1, (0, 255, 255), 2)
        
        c = max(contours, key = cv2.contourArea)
        m = cv2.moments(c)
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        center.append(cX)
        center.append(cY)
        print("Center of object (x,y): (" + str(cX) + ", " + str(cY) + ")")
        return center
        #print center
        #print "Center of object (x,y): (", cX, ", ", cY, ")"
        
        #Draw circle and bounding box to indicate center of object
        #cv2.circle(output, (cX, cY), 2, (0, 255, 255), -1)
        #x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Show new image (cv2.imshow() does not work on macOS)
    #cv2.imshow("images", np.hstack([img, output]))
    #cv2.waitKey(0)



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
    print(temp, "->", centerY)
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

print(dataNew1[0])
print(dataNew2[0])

    

for i in range(0, len(dataPos) - seq_length):
    _x = dataNew1[i:i + seq_length]
    _y = dataNew2[i + seq_length] #Next closest angles 
    print(_x, "->", _y)
    dataPosX.append(_x)
    dataPosY.append(_y)


#Train/test split
train_size = int(len(dataPosY) * 0.95)
test_size = len(dataPosY) - train_size
trainX, testX = np.array(dataPosX[0:train_size]), np.array(
    dataPosX[train_size:len(dataPosX)])
trainY, testY = np.array(dataPosY[0:train_size]), np.array(
    dataPosY[train_size:len(dataPosY)])
#np.random.shuffle(trainX)
#np.random.shuffle(trainY)
print(testX)
print(testY)


#Input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

#Build LSTM Network
#Minimize amount of hidden layers because of training time 
cell = tf.nn.rnn_cell.LSTMCell(
        num_units = hidden_dim, state_is_tuple = True, activation = tf.nn.softmax)
multi_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype = tf.float32)
print (outputs[:, -1])
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
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter("output", sess.graph)
    
    #Training
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict = {
            X: trainX, Y: trainY})
        print("[Step: {}] Loss: {}".format(i, step_loss))
        test_predict = sess.run(Y_pred, feed_dict = {X: testX})
        rmse_val = sess.run(rmse, feed_dict = {
        targets: testY, predictions: test_predict})
        
    writer.close()

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
    print('Predicting position...')
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
    print("Actual: {}".format(testX))
    
    test_predict = sess.run(Y_pred, feed_dict = {X: testX})
    rmse_val = sess.run(rmse, feed_dict = {
        targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
    print("Prediction: {}".format(test_predict))

    errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToPos', [], [0.85, -0.8, 0.78], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    time.sleep(2)
    errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'activateVacuum', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    time.sleep(2)
    errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'goToDropPos', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
    time.sleep(2)
    errorCode, outInts, outFloats, outStrings, outBuffer = vrep.simxCallScriptFunction(clientID, 'Baxter_leftArm_joint1', vrep.sim_scripttype_childscript, 'deactivateVacuum', [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)  

    #Plot results
    #plt.plot(testY)
    #plt.plot(test_predict)
    #plt.show()

    saver.save(sess, './trained_model', global_step = iterations)
        
    #Fix saving model. Parameters are not saving(?). Restored model does not predict correctly
