# Chapter 1 HandTracking Basics
import cv2
import mediapipe as mp
import time
cap=cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime = 0 #previousTime
cTime = 0 #currentTime

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
       for handLms in results.multi_hand_landmarks:
           for id,lm in enumerate(handLms.landmark):   #id is index number of finger landmarks
            #print(id,lm)
            h , w, c = img.shape
            cx, cy = int(lm.x*w),int(lm.y*h)#cx and cy is position of the center(converting in pixels because we need it it pixels)
            print(id,cx,cy)
            #if id==8:
            cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)


           mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) #drawing hands
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,
                3,(255,0,255),3)

# we can draw connections as well
    cv2.imshow("Image",img)
    cv2.waitKey(1)

# now we are going to create a module out of this

