import numpy as np
import argparse 
import os
import cv2
from matplotlib import pyplot as plt

### Leer argumentos de entrada ###
ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True, help = "path of image")
ap.add_argument("-covers", required = True, help = "path of database")

args = vars(ap.parse_args())

query = args['query']
database = args['covers']

# Detector
orb = cv2.ORB_create()


#lectura y detección de la imagen principal
img1 = cv2.imread(query,0)
kp1, des1 = orb.detectAndCompute(img1,None)

maxNumMatches = 0
minMatches = 20
maxMatches = None
maxImg = None
maxKp = None


for img in os.listdir(database):
    #Lectura de la segunda imagen
    img = cv2.imread(database+"/"+img,0) 

    # keypoints de la segunda imagene
    kp2, des2 = orb.detectAndCompute(img,None)

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match
    matches = bf.match(des1,des2)

    aux = len(matches)

    if((aux>minMatches) & (aux>maxNumMatches)):
        maxNumMatches = aux
        maxKp = kp2
        maxMatches = matches
        maxImg = img

if (maxNumMatches == 0):
    print ("Ninguna coincidencia")
else:
    # ordenación por distancia
    maxMatches = sorted(maxMatches, key = lambda x:x.distance)

    # Se dibujan los primeros x matches.
    img3 = None
    img3 = cv2.drawMatches(img1,kp1,maxImg,maxKp,maxMatches,img3, flags=2)

    plt.imshow(img3),plt.show()
    cv2.waitKey(0) 