import cv2
import mediapipe as mp
import time
import math
import numpy as np
from tensorflow.keras.models  import load_model

#%%

drawstick = 8
drawcolor = (25,100,50)

def calculateDistance(x1=0,y1=0,x2=1000,y2=1000):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print("distance : ", dist)
    return dist


def draw(drawpoints):
    for point in drawpoints:
        cv2.circle(img, (point[0], point[1]), drawstick, drawcolor, cv2.FILLED)
        cv2.circle(recognition, (point[0], point[1]), drawstick, drawcolor, cv2.FILLED)




#find contours on image and draw it.
def findText(img):
    #perform some basic operation to smooth image
    img = img[:,50:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #find threshold image
    ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)
    ctrs, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        #draw rectangle on image using contours
        
        area = rect[2]*rect[3]
        if area>5000:
            cv2.rectangle(recognition, (rect[0]+50, rect[1]), (rect[0] + rect[2]+50, rect[1] + rect[3]), (0, 255, 0), 3)
            leng = int(rect[3] * 1.6)
            pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
            pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))

            #resize image
            roi = img[pt1:pt1+leng, pt2:pt2+leng]
            roi = cv2.resize(roi,(28, 28), interpolation=cv2.INTER_AREA)
        
            #reshape your image according to your model
            roi = roi.reshape(-1,28, 28, 1)
            roi = np.array(roi, dtype='float32')
            roi /= 255
            #to perform prediction on your image
            pred_array = model.predict(roi)
            pred_array = np.argmax(pred_array)
        
            #print result
            print('Result: {0}'.format(pred_array))
        
            #print text on your image
            cv2.putText(recognition, str(pred_array), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
            #show your image



"""
def draw(Tempdrawpoints):
    list = []
    for point in Tempdrawpoints:
        list.append([point[0], point[1]])
    if len(list)>1:
        list = np.array(list, np.int32)
        cv2.polylines(img, pts=[list], isClosed=False, color=(100,100,200), thickness=3)
"""
#%%
#model = load_model('D:/keras_model.h5')
model = load_model("MNIST-CNN.model")

#%%
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
pTime = 0
cTime = 0



drawPoints = []
TempdrawPoints = []
switch = 0


while True:
    ret, img = cap.read()
    if ret:
        img = cv2.flip(img, 1)
        recognition = np.zeros((img.shape), np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)
        cv2.rectangle(recognition, (0,0), (50, 480), (100, 120, 0), 3)
        cv2.rectangle(img, (0,0), (50, 480), (100, 120, 0), 3)
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                Temp = [0,0,0,0]
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(img, str(i), (xPos-25,yPos-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    print(i, xPos, yPos)
                    
                    if i == 4:
                        cv2.circle(img, (xPos,yPos), 10, (255,0,0), cv2.FILLED)
                        Temp[0] = xPos
                        Temp[1] = yPos
                    
                    if i ==8:
                        cv2.circle(img, (xPos,yPos), 10, (255,0,0), cv2.FILLED)
                        Temp[2] = xPos
                        Temp[3] = yPos
                        
                        if calculateDistance(Temp[0],Temp[1],Temp[2],Temp[3])<50:
                            switch = 1
                            TempdrawPoints.append([int((Temp[0]+Temp[2])/2),int((Temp[1]+Temp[3])/2)])
                            #drawPoints.append([xPos,yPos])
                            if (int((Temp[0]+Temp[2])/2))<50 :
                                TempdrawPoints = []
                                
                        else:
                            switch = 0
                
                
        #畫出點
        draw(TempdrawPoints)
        findText(recognition)
        
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('img', img)
        cv2.imshow('recognition', recognition)
        
        print("-----------switch:", switch)
        
    if cv2.waitKey(1) == ord('q'):
        break