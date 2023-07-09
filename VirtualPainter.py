import cv2
import time
import numpy as np
import os 
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(folderPath + "/" + imPath)
    # print(folderPath + "/" + imPath)
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (0,0,255)
# -------------------------------------
brushThickness = 8
eraserThickness = 90
# -------------------------------------
xp,yp=0,0

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetection(detectionCon=0.65,maxHands=1)
imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
        
    # Find Hand Landmarks
    img = detector.findhands(img)
    lmList = detector.findPosition(img,draw=False)
    
    if len(lmList) != 0:
        # print(lmList)

        # Tip of Index Finger and Middle Finger
        x1,y1 = (lmList[8][1:])
        x2,y2 = (lmList[12][1:])
        
        # Checking which Fingers are UP
        fingers = detector.fingersUp()
        # print(fingers)
        
        # If Selection Mode - Two Finger are Up
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            print("Selection Mode")
            # Checking for the Click
            if y1 < 125:
                if 150<x1<280:
                    header= overlayList[0]
                    drawColor = (0,0,255)
                elif 320<x1<500:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 635<x1<720:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 855<x1<940:
                    header = overlayList[3]
                    drawColor = (105,105,105)
                elif 1075<x1<1280:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img,(x1,y1-40),(x2,y2+40),drawColor,cv2.FILLED)
                    
        
        # If Drawing Mode -Index Finger is Up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing Mode")
            
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:  
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp = x1,y1
            
        # Clear Canvas when all Fingers are Up
        # if all (x>=1 for x in fingers):
        #     imgCavas = np.zeros((720,1280,3),np.uint8)
            
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInverse = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInverse)
    img = cv2.bitwise_or(img,imgCanvas)
    
    #Setting the Header Image
    img[0:125,0:1280] = header
    # img = cv2.addWeighted(img,0,5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    # cv2.imshow("Canvas",imgCanvas)
    # cv2.imshow("Inverse",imgInverse)
    cv2.waitKey(1)