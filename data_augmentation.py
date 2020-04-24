
from PIL import Image, ImageFilter
import sys
import random
import argparse 
import os


### Leer argumentos de entrada ###
ap = argparse.ArgumentParser()
ap.add_argument("-i","--input_dataset", required = True, help = "path of the input dataset")
ap.add_argument("-f","--factor", required = True, help = "factor")
ap.add_argument("-o","--output_dataset", required = True, help = "path of the output dataset")

args = vars(ap.parse_args())

inputURL = args['input_dataset']
factor = int(args['factor'])
outputURL = args['output_dataset']

cont = 0

for i in range(factor):
    ### Leer imagenes de un directorio
    for img in os.listdir(inputURL):
        #print (img)
        b = random.choice([True, False])
        res = random.choice([True, False])
        rot = random.choice([True, False])
        imagen = Image.open(inputURL+"/"+img)

        if(b==True):
            factorBlur = random.randint(2, 10)
            imagen = imagen.filter(ImageFilter.GaussianBlur(factorBlur))

        if(res==True):
            factorRes = random.uniform(0.25, 2.5)
            imagen = imagen.resize((int(imagen.width*factorRes), int(imagen.height*factorRes)))

        if(rot==True):
            factorRot = random.randint(0,6)
            if(factorRot == 0):
                imagen = imagen.transpose(Image.FLIP_LEFT_RIGHT)
            if(factorRot == 1):
                imagen = imagen.transpose(Image.FLIP_TOP_BOTTOM)
            if(factorRot == 2):
                imagen = imagen.transpose(Image.ROTATE_90)
            if(factorRot == 3):
                imagen = imagen.transpose(Image.ROTATE_180)
            if(factorRot == 4):
                imagen = imagen.transpose(Image.ROTATE_270)
            if(factorRot == 5):
                imagen = imagen.transpose(Image.TRANSPOSE)

        try:
            os.stat(outputURL)
        except:
            os.mkdir(outputURL)

        imagen.save(outputURL+"/"+str(cont)+img)

        cont=cont+1
