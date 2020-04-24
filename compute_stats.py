import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import argparse

## Leer argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument("--inference", required=True, help="Path of detection data")
ap.add_argument("--groundtruth", required=True, help="Path of groundtruth")
ap.add_argument("--output_graphs", required=True, help="Output path")
args = vars(ap.parse_args())

URLDetection = args['inference']
URLGroundtruth = args['groundtruth']
URLoutput = args['output_graphs']


detection = pd.read_csv(URLDetection, na_values="-")
groundtruth = pd.read_csv(URLGroundtruth, na_values="-")

detection_values = detection.values
groundtruth_values = groundtruth.values

# Creación de las columnas
ranges = range(0, 250, 50)
thresholds_length = 5
intervalos = [[ranges[n], ranges[n+1]] for n in range(thresholds_length) if n < thresholds_length-1]
intervalos.append([ranges[thresholds_length-1], math.inf])

labels = ["[" + str(intervalos[n][0]) + ", " + str(intervalos[n][1]) + ")" for n in range(len(intervalos))]
labels.append("Errors")


#Area 2D
a = detection_values[:, 1] - groundtruth_values[:, 1]
total_values = len(a)
error_count = np.sum(np.isnan(a))
a = np.abs(np.where(np.isfinite(a), a, 0))
results = [(np.count_nonzero((a >= intervalos[n][0]) & (a < intervalos[n][1]))/total_values)*100 for n in range(len(intervalos))]
results.append(error_count)


#Area  3D
a = detection_values[:, 2] - groundtruth_values[:, 2]
total_values = len(a)
error_count = np.sum(np.isnan(a))
a = np.abs(np.where(np.isfinite(a), a, 0))
results2 = [(np.count_nonzero((a >= intervalos[n][0]) & (a < intervalos[n][1]))/total_values)*100 for n in range(len(intervalos))]
results2.append(error_count)


#Creación de las columnas de complexity
ranges2 = range(5)
thresholds2_length = 5
intervalos2 = [[ranges2[n], ranges2[n+1]] for n in range(thresholds2_length) if n < thresholds2_length-1]
intervalos2.append([ranges2[thresholds2_length-1], math.inf])

labels = ["[" + str(intervalos2[n][0]) + ", " + str(intervalos2[n][1]) + ")" for n in range(len(intervalos2))]
labels.append("Errors")

#Calculo de los valores de complexity
a = detection_values[:, 3] - groundtruth_values[:, 3]
total_values = len(a)
error_count = np.sum(np.isnan(a))
a = np.abs(np.where(np.isfinite(a), a, 0))
results3 = [(np.count_nonzero((a >= intervalos[n][0]) & (a < intervalos[n][1]))/total_values)*100 for n in range(len(intervalos))]
results3.append(error_count)

# visualzación de los datos
plt.title("Area 2D")
plt.xlabel("Squared feet error")
plt.ylabel("Percentage of blueprints")
plt.bar(range(len(results)), results, color=['red','red','red','red','red','black'])
plt.xticks(range(len(results)), labels)
plt.ylim([0, 100])
plt.savefig(URLoutput + "/Area2D.png")
plt.show()

plt.title("Area 3D")
plt.xlabel("Squared feet error")
plt.ylabel("Percentage of blueprints")
plt.bar(range(len(results2)), results2, color=['red','red','red','red','red','black'])
plt.xticks(range(len(results2)), labels)
plt.ylim([0, 100])
plt.savefig(URLoutput + "/Area2D.png")
plt.show()

plt.title("complexity")
plt.xlabel("Inches per foot")
plt.ylabel("Percentage of blueprints")
plt.bar(range(len(results3)), results3, color=['red','red','red','red','red','black'])
plt.xticks(range(len(results3)), labels)
plt.ylim([0, 100])
plt.savefig(URLoutput + "/Area2D.png")
plt.show()