
import sys
import argparse 
import os
import cv2
import  numpy as np

### Leer argumentos de entrada ###
ap = argparse.ArgumentParser()
ap.add_argument("-images", required = True, help = "path of the input dataset")
ap.add_argument("-out", required = True, help = "path of result")

args = vars(ap.parse_args())

inputURL = args['images']
outputURL = args['out']

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

image = cv2.imread(inputURL+'/'+os.listdir(inputURL)[0])
(h, w) = np.shape(image)[:2]

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')
out = cv2.VideoWriter(outputURL,fourcc, 24.0, (w,h))

for img in os.listdir(inputURL):
    imagen = cv2.imread(inputURL+"/"+img)
    rects, weights = hog.detectMultiScale(imagen, winStride=(8, 8), padding=(32, 32), scale=1.05)
    
    for r in rects:
        cv2.rectangle(imagen,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),3)

    
    cv2.imshow('frame',imagen)
    out.write(imagen)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()