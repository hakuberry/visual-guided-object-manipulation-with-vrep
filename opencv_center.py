import cv2
import numpy as np

image_dir = "D:/Purdue/Industrial Robotics and Flexible Assembly/Project/Source Code/Training Data/grid_1_5.png"



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
    cv2.drawContours(output, contours, -1, (0, 255, 255), 2)
        
    c = max(contours, key = cv2.contourArea)
    m = cv2.moments(c)
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])
    center.append(cX)
    center.append(cY)
    print("Center of object (x,y): (" + str(cX) + ", " + str(cY) + ")")
    print (center)
        
    #Draw circle and bounding box to indicate center of object
    cv2.circle(output, (cX, cY), 2, (0, 255, 255), -1)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

#Show new image (cv2.imshow() does not work on macOS)
cv2.imshow("images", np.hstack([img, output]))
cv2.waitKey(0)

    
