import cv2 as cv
from cvzone.HandTrackingModule import HandDetector 
import numpy as np 
import math
import time

# Initialize video capture
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize hand detector
detector = HandDetector(maxHands=1) 

#variables
offset = 20
imgSize = 300
counter = 0

#folder that stores data
folder = 'data/ily'

while True:
    success , img = capture.read()
    hands , img = detector.findHands(img)
    if hands :
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #imgcrop: cropped image of just hands
        #imgwhite : so that all images can have equal dimensions (the image that gets stored in folder) 
        imgWhite = np.ones((imgSize , imgSize ,3) , np.uint8)*255 #colour range from 0 -255
        imgcrop = img[y-offset:y+h+offset , x-offset:x+w+offset] #starting and ending height of box (y,x)
        
        #putting the cropped box on white box 
        #and to centre the croped box on white box:
        aspectRatio = h/w
        if aspectRatio > 1:#i.e height is bigger
            k = imgSize / h #size of white box (strech the height to size of boz)
            wcal = math.ceil(k*w) #if 3.4 or 3.5 --->4 (strech width to remaining of the box)
            imgrezise = cv.resize(imgcrop,(wcal , imgSize))
            imgreziseshape = imgrezise.shape
            #widht gap
            wgap = math.ceil((imgSize - wcal)/2)
            imgWhite[: , wgap:wcal+wgap] = imgrezise

        else:#i.e width is bigger
            k = imgSize/w 
            hcal = math.ceil(k*h) 
            imgrezise = cv.resize(imgcrop,(imgSize , hcal))
            imgreziseshape = imgrezise.shape
            #widht gap
            hgap = math.ceil((imgSize - hcal)/2)
            imgWhite[hgap:hcal+hgap , :] = imgrezise

        #show the croped image and white image
        cv.imshow("image", imgcrop)
        cv.imshow('imgwhite' , imgWhite)
    
    cv.imshow("camera" , img)

    #reads the key pressed 
    #if 's' is pressed , will write the white image to the desired('A') folder
    #if 'q' is pressed , break the loop
    key = cv.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print("image",counter)
    if key == ord("q"):
        break

#release the capture variable and destroy all the windows 
capture.release()
cv.destroyAllWindows()