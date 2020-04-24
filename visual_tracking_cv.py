import numpy as np
import argparse 
import os
import cv2
from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("--image", required = True, help = "path of video")

args = vars(ap.parse_args())

videoURL = args['image']


#Inicio del programa
seleccion = True 
seleccionado = False
#Primer frame del video
hsv = None 
#Posicion del elemento que queremos seguir
roiPos = [] 

#Variables para el marcado de movimiento
font = cv2.FONT_HERSHEY_SIMPLEX


#variables para Camshift
track_window = []
term_crit =(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)



cap = cv2.VideoCapture(videoURL)
while(cap.isOpened()):

    #Obtenemos cada frame del video
    ret, frame_video = cap.read()
    if ret == True:
        #Pasamos de BGR a HSV
        hsv = cv2.cvtColor(frame_video,cv2.COLOR_BGR2HSV)

        #Comprobamos si estamos en el momento de seleccion
        while(seleccion):
            cv2.imshow("Frame", frame_video)
            roiPos = cv2.selectROI(frame_video,True)
            track_window.append(roiPos[0])
            track_window.append(roiPos[1])
            track_window.append(roiPos[2])
            track_window.append(roiPos[3])

            roi = frame_video[roiPos[1]:roiPos[1]+roiPos[3], roiPos[0]:roiPos[0]+roiPos[2]]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv_roi,np.array((0., 60., 32.,)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask2,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

            seleccionado = True
            seleccion = False

        if (seleccionado):
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret,track_window = cv2.CamShift(dst, (track_window[0],track_window[1],track_window[2],track_window[3]), term_crit)
            res = cv2.bitwise_and(frame_video,frame_video)

            #Mostrar por pantalla CamShift
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(res,[pts],True,255,2)

            cv2.imshow("Frame", img2)
            if cv2.waitKey(24) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()


        