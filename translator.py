import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier 
import numpy as np 
import math
import pyttsx3
import threading

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1) 
classifier = Classifier('model/keras_model.h5' , "model/labels.txt")

offset = 20
imgSize = 300
counter = 0
folder = 'data/C'
labels = ['A' , 'B' , 'C' ,'i like you']
engine = pyttsx3.init()

# Function for text-to-speech
def speak_label(label):
    #engine.setProperty('rate' ,50)
    engine.say(label)
    engine.runAndWait()

while True:
    success , img = capture.read()
    imgo = img.copy()
    hands , img = detector.findHands(img)
    if hands :
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize , imgSize ,3) , np.uint8)*255 #colour range from 0 -255
        imgcrop = img[y-offset:y+h+offset , x-offset:x+w+offset] #starting and ending height of box (y,x)
        imgCropShape = imgcrop.shape

        #putting the cropped box on white box 
        #to centre the croped box on white box:
        aspectRatio = h/w
        if aspectRatio > 1:#i.e height in bigger
            k = imgSize / h #size of white box (strech the height to size of boz)
            wcal = math.ceil(k*w) #if 3.4 or 3.5 --->4 (strech widht to remaining of the box)
            imgrezise = cv.resize(imgcrop,(wcal , imgSize))
            imgreziseshape = imgrezise.shape
            #widht gap
            wgap = math.ceil((imgSize - wcal)/2)
            imgWhite[: , wgap:wcal+wgap] = imgrezise
            
        else:
            k = imgSize/w
            hcal = math.ceil(k*h)
            imgrezise = cv.resize(imgcrop,(imgSize , hcal))
            imgreziseshape = imgrezise.shape
            hgap = math.ceil((imgSize - hcal)/2)
            imgWhite[hgap:hcal+hgap , :] = imgrezise

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        identified_label = labels[index]
        #print(prediction, index)

        # Start a new thread for speaking the identified label
        threading.Thread(target=speak_label, args=(identified_label,)).start()
        
        #displaying the identified word
        cv.rectangle(imgo, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv.FILLED)
        cv.putText(imgo, labels[index], (x, y - 30), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv.rectangle(imgo, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
        cv.imshow("image", imgcrop)
        cv.imshow('imgwhite' , imgWhite)
    
    cv.imshow("camera" , imgo)
    key = cv.waitKey(1)
    
    if key == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
