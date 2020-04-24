
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

### Leer argumentos de entrada ###
ap = argparse.ArgumentParser()
ap.add_argument("--image", help = "path of image or video")
ap.add_argument("--video", help = "path of image or video")

args = vars(ap.parse_args())

if(args['image']):
	URL = args['image']

if(args['video']):
	URL = args['video']

width = 320
height = 320
padding = 0.0
min_conf = 0.5
close = False


URLEast = "./frozen_east_text_detection.pb"

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		for x in range(0, numCols):
			if scoresData[x] < min_conf:
				continue

			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)


def detector (image):
	global close
	# Cargar la imagen y redimensionar
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	(newW, newH) = (width, height)
	rW = origW / float(newW)
	rH = origH / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# carga de la red preentrenada East
	net = cv2.dnn.readNet(URLEast)

	# crear blob de la imagen
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# obtener los resultados
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# inicializacion de la lista de resultados
	results = []

	# loop sobre la localizacion de los resultados
	for (startX, startY, endX, endY) in boxes:
		# posicionamiento
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# padding sobre el resultado para mejorar la localización (caja mas grande)
		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)

		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		# zona de interes
		roi = orig[startY:endY, startX:endX]

		# configuracion de tesseract
		# 1  idioma
		# 2  flag de OEM, red que se quiere utilizar
		# 3  valor de OEM
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		# almacenamiento de los resultados
		results.append(((startX, startY, endX, endY), text))

	# se recorren los resultados
	for ((startX, startY, endX, endY), text) in results:

		# se muestra por pantalla los resultados
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		output = orig.copy()
		cv2.rectangle(output, (startX, startY), (endX, endY),(226, 43, 138), 2)
		cv2.putText(output, text, (startX, startY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (226, 43, 138), 3)

		cv2.imshow("Resultado", output)

		if(args['image']):
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break

		if(args['video']):
			if cv2.waitKey(1) & 0xFF == ord('q'):
				close = True
				break
			


if(args['image']):
	img = cv2.imread(URL)
	detector(img)
	cv2.destroyAllWindows()


elif(args['video']):
	cap = cv2.VideoCapture(URL)

	while(cap.isOpened()):
		ret, frame = cap.read()
		if(ret == True):
			if(close == False):
				detector(frame)
			else:
				break

	cap.release()
	cv2.destroyAllWindows()

else:
	print("Se debe introducir una imagen o un video a analizar")