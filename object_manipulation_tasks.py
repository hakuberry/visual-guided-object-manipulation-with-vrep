import tensorflow as tf
import numpy as np
import csv
import os
import cv2
import time
import random
import matplotlib
import matplotlib.pyplot as plt

tf.set_random_seed(777)

#Training set directory
training_dir = "D:/Purdue/Industrial Robotics and Flexible Assembly/Project/Source Code/Training Data/time_series.csv"


def MinMaxNormalize(data):
    num = data - np.min(data, 0)
    den = np.max(data, 0) - np.min(data, 0)
    return num / (den + 1e-7)

def GetObjectCenter(image_dir):
    img = cv2.imread(image_dir, 1)
    center = []
    
    #Lower and upper bound for color (B,G,R)
    lower = [17, 5, 90]
    upper = [80, 60, 255]

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
        

        #x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Show new image (cv2.imshow() does not work on macOS)
    #cv2.imshow("images", np.hstack([img, output]))
    #cv2.waitKey(0)



#Training parameters
seq_length = 1
data_dim = 4
hidden_dim = 4
output_dim = 2
learning_rate = 0.01
iterations = 50000

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

#Build the dataset
dataObjectX = []
dataObjectY = []
dataAnglesX = []
dataAnglesY = []
dataX = []
dataY = []
dataTemp = []
dataNew1 = []
dataNew2 = []

dataObject = np.loadtxt(training_dir, dtype = 'str', delimiter = ',', skiprows = 1, usecols = [0])
#print(dataObject)
dataAngles = np.loadtxt(training_dir, delimiter = ',', skiprows = 1, usecols = range(1, 3))
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
    temp1.append(dataAngles[i][0])
    temp1.append(dataAngles[i][1])

    temp2.append(dataAngles[i][0])
    temp2.append(dataAngles[i][1])

    dataNew1.append(temp1)
    dataNew2.append(temp2)

print(dataNew1[0])
print(dataNew2[0])

    

for i in range(0, len(dataAngles) - seq_length):
    _x = dataNew1[i:i + seq_length]
    _y = dataNew2[i + seq_length] #Next closest angles 
    print(_x, "->", _y)
    dataAnglesX.append(_x)
    dataAnglesY.append(_y)


#Train/test split
train_size = int(len(dataAnglesY) * 0.95)
test_size = len(dataAnglesY) - train_size
trainX, testX = np.array(dataAnglesX[0:train_size]), np.array(
    dataAnglesX[train_size:len(dataAnglesX)])
trainY, testY = np.array(dataAnglesY[0:train_size]), np.array(
    dataAnglesY[train_size:len(dataAnglesY)])
print(testX)
print(testY)


#Input placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 2])

#Build LSTM Network
#Minimize amount of hidden layers because of training time on laptop
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
targets = tf.placeholder(tf.float32, [None, 2])
predictions = tf.placeholder(tf.float32, [None, 2])
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
        print("RMSE: {}".format(rmse_val))
        if rmse_val < 0.2:
            break

    writer.close()

    
    #Test
    test_predict = sess.run(Y_pred, feed_dict = {X: testX})

    rmse_val = sess.run(rmse, feed_dict = {
        targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
    print(testY)
    print(test_predict)

    plt.plot(rmse_val)
    plt.show()
    

    
    
    #Fix saving model. Parameters are not saving(?). Restored model does not predict correctly
    #saver.save(sess, '/Volumes/CRUZER U/vgomt_2.ckpt', global_step = iterations)
    #saver.export_meta_graph(filename="/Volumes/CRUZER U/vgomt_2")

    #plt.plot(testY)
    #plt.plot(test_predict)
    #plt.show()

