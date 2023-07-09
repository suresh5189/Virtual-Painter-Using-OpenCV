import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetection():
    def __init__(self,mode=False,maxHands=2,model_complexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils 
        self.tipIds = [4,8,12,16,20]
 
#  -----------------------------------------------------------------------------------------       
    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # print(imgRGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks) 
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:    
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
# ------------------------------------------------------------------------------------------
    
    def findPosition(self,img,handNo=0,draw=True):
        bbox = []
        self.lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lms in enumerate(myHand.landmark):
                # print(id,lms)
                h,w,c = img.shape
                cx,cy = int(lms.x*w),int(lms.y*h)
                # print(id,cx,cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)
                    
        return self.lmList

 # -------------------------------------------------------------------------------------------
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] >  self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        
        # 4 Fingers 
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
# ---------------------------------------------------------------------------------------------
    
    def findDistance(self,p1,p2,img,draw=True):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(cx,cy),8,(0,0,255),cv2.FILLED)
            length = math.hypot(x2-x1,y2-y1)
            
        return length, img, [x1,y1,x2,y2,cx,cy]
        
# ----------------------------------------------------------------------------------------------
        
    
def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetection()

    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[0])
            
        currentTime = time.time()
        FPS = 1/(currentTime-previousTime)
        previousTime = currentTime
        
        cv2.putText(img,str(int(FPS)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        cv2.imshow("Output", img)
        cv2.waitKey(1)
    
    
    
    
if __name__ == "__main__":
    main()