import matplotlib.pyplot as plt
import pandas as pd
import ordpy
import cv2
import glob
import os

def plot_HC_plan():
    data = []
    folder = os.getcwd() + "/data/*.*"
    for file in glob.glob(folder):
        image_read = cv2.imread(file)
        # conversion numpy array into rgb image to show
        img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
        data.append(ordpy.complexity_entropy(img, dx = 3, dy = 3, taux = 1, tauy = 1))

    data = pd.DataFrame(data)
    plt.scatter(data[0], data[1])
    plt.show()

plot_HC_plan()