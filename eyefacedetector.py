import argparse 
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-video", required = True, help = "path to where the video file resides")
ap.add_argument("-out", required = True, help = "path to the output video")

args = vars(ap.parse_args())

videoURL = args['video']
outputURL = args['out']


cap = cv2.VideoCapture(videoURL)

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')
out = cv2.VideoWriter(outputURL,fourcc, 24.0, (w,h))

while(cap.isOpened()):   
    ret, frame = cap.read()   
    if(ret == True):
        faceCascade = cv2.CascadeClassifier("./eye_face/haarcascade_eye.xml")
        rects = faceCascade.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors =5,minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)  
        
        faceCascade = cv2.CascadeClassifier("./eye_face/haarcascade_frontalface_default.xml")
        rects2 = faceCascade.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors =5,minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
        
        for r in rects:
            cv2.rectangle(frame,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),3)

        for r in rects2:
            cv2.rectangle(frame,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),3)
        
        cv2.imshow('frame',frame)
        out.write(frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()