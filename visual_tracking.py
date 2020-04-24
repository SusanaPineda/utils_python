#import sys
import random
import argparse 
#import os
from collections import deque
import cv2
import numpy as np

### Leer argumentos de entrada ###
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", required = True, help = "path of the video")
ap.add_argument("-m","--min_values", required = True, help = "min values of HSV", nargs="+", type=int)
ap.add_argument("-mx","--max_values", required = True, help = "max values of HSV", nargs="+", type=int)
ap.add_argument("-0","--output", required = False, help = "path to save the video")

args = vars(ap.parse_args())

inputURL = args['video']
minValues = np.array(args['min_values'])
maxValues = np.array(args['max_values'])
outputURL = args['output']

kernel = np.ones((3,3),np.uint8)

pts = deque(maxlen=10)

cap = cv2.VideoCapture(inputURL)

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

if outputURL:
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(outputURL, fourcc, 24.0, (w,h))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #Filtro gaussiano
        fm = frame.copy()
        frame = cv2.GaussianBlur(frame,(5,5),0)
        
        #Paso a HSV
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        #Segmentacion por color
        mask = cv2.inRange(frame, minValues, maxValues)
        
        #Aplicacion de la mascara
        frame = cv2.bitwise_and(frame,frame, mask= mask)       
        
        #Erosion
        frame = cv2.erode(mask,kernel,iterations = 2) 
        
        #Dilatacion
        frame = cv2.dilate(mask,kernel,iterations = 2)
        
        #Deteccion de contornos
        contours, hierachy =cv2.findContours(frame, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        #contorno de mayor area
        max_area = 0
        for c in contours:
            aux = cv2.contourArea(c)
            if(max_area < aux):
                max_area = aux
                pos = c

        #Centro del contorno mayor
        M = cv2.moments(array=pos)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #Pintado por pantalla
        #circulo minimo rojo
        ((x, y), radius) = cv2.minEnclosingCircle(pos)
        cv2.circle(fm, (int(x),int(y)),int(radius),(0,0,255), 2)
        #centroide verde
        cv2.circle(fm, (cx,cy),5,(0,255,0),-1)

        #trayectoria azul
        pts.appendleft((cx,cy))
        for i in range(1, len(pts)):
            if((pts[i-1] != None) and (pts[i] != None)):
                thickness = np.sqrt(10/float(i+1))*2
                cv2.line(fm, pts[i - 1], pts[i], (255, 0, 0), int(thickness))

        
        #Mostrar por pantalla
        cv2.imshow("Frame", fm)

        #Escritura
        if(outputURL):
            out.write(fm)
        #Control FR
        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
if(outputURL):
    out.release()
cv2.destroyAllWindows()