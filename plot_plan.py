import matplotlib.pyplot as plt
import pandas as pd
import ordpy
import cv2
import glob
import os
import seaborn as sns
import numpy as np

def process_data():
    data = []
    data_fisher = []
    folder = os.getcwd() + "/data/images/*.*"
    for file in glob.glob(folder):
        img_id = int(file.split("/")[-1].split(".")[0])
        image_read = cv2.imread(file)
        # conversion numpy array into rgb image to show
        img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)   
        #Laplacian filter 
        img = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=5))   
        #
        hc = ordpy.weighted_complexity_entropy(img, dx = 2, dy = 2, taux = 1, tauy = 1, q=5)
        data.append([img_id, hc[0], hc[1]])
        fs = ordpy.weighted_fisher_shannon(img, dx = 2, dy = 2, taux = 1, tauy = 1, q=5)
        data_fisher.append([img_id, fs[0], fs[1]])

    data = pd.DataFrame(data, columns = ["img_id", "Entropy", "Complexity"]).sort_values("img_id").reset_index(drop = True)
    data_fisher = pd.DataFrame(data_fisher, columns = ["img_id", "Entropy", "Fisher information"]).sort_values("img_id").reset_index(drop = True)
    labels = pd.read_csv("data/labels/groups_DPD.txt", header = None)
    data["Label"] = labels
    data_fisher["Label"] = labels
    dict_label = {"consolidacao": "Pulmonary Consolidation",
                  "enfisema": "Emphysematous Area",
                  "espessamento": "Septal Thickening",
                  "favo_de_mel": "Honeycomb",
                  "normal": "Healthy",
                  "vidro_fosco": "Ground-glass Opacity"}
    data.Label = data.Label.apply(lambda x : dict_label[x] if x in dict_label.keys() else x)
    data_fisher.Label = data_fisher.Label.apply(lambda x : dict_label[x] if x in dict_label.keys() else x)    
    # data.to_csv("weighted_HC_data.csv", index = False)
    # data_fisher.to_csv("weighted_FS_data.csv", index = False)
    return data, data_fisher

def process_bandt_pomp():
    data = []
    data_fisher = []
    folder = os.getcwd() + "/data/images/*.*"
    for file in glob.glob(folder):
        img_id = int(file.split("/")[-1].split(".")[0])
        image_read = cv2.imread(file)
        # conversion numpy array into rgb image to show
        img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
        #Laplacian filter
        img = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=3))

        ord_dis = weighted_ordinal_distribution(img, dx = 2, dy = 2, taux = 1, tauy = 1, q = 5)
        data.append([img_id] + list(ord_dis[1]))

    data = pd.DataFrame(data).sort_values(0).reset_index(drop = True)
    labels = pd.read_csv("data/labels/groups_DPD.txt", header = None)
    data["Label"] = labels
    dict_label = {"consolidacao": "Pulmonary Consolidation",
                  "enfisema": "Emphysematous Area",
                  "espessamento": "Septal Thickening",
                  "favo_de_mel": "Honeycomb",
                  "normal": "Healthy",
                  "vidro_fosco": "Ground-glass Opacity"}
    data.Label = data.Label.apply(lambda x : dict_label[x] if x in dict_label.keys() else x)
    data.to_csv("weighted_BP_data_(5).csv", index = False)
    return data

def process_smoothness_structure():
    data = []
    all_folders = os.getcwd() + "/data/Pulmonar/*"
    for folder in glob.glob(all_folders):
        label = folder.split("/")[-1]
        folder += "/*.*"
        for file in glob.glob(folder):
            img_id = int(file.split("/")[-1].split(".")[0])
            image_read = cv2.imread(file)
            # conversion numpy array into rgb image to show
            img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
            # Laplacian filter
            img = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=5)).astype('float64')
            # Tiny random noise to avoid zero variance 
            ord_dis = ordpy.weighted_smoothness_structure(img, q = 0.5)
            data.append([img_id, ord_dis[0], ord_dis[1], label])

    # data = pd.DataFrame(data).sort_values(0).reset_index(drop = True)
    data = pd.DataFrame(data, columns = ["img_id", "Smoothness", "Curve structure", "Label"]).sort_values('img_id').reset_index(drop = True)

    # labels = pd.read_csv("data/labels/groups_DPD.txt", header = None)
    # data["Label"] = labels
    dict_label = {"consolidacao": "Pulmonary Consolidation",
                  "enfisema": "Emphysematous Area",
                  "espessamento": "Septal Thickening",
                  "favo_de_mel": "Honeycomb",
                  "normal": "Healthy",
                  "vidro_fosco": "Ground-glass Opacity"}
    data.Label = data["Label"].apply(lambda x : dict_label[x] if x in dict_label.keys() else x)

    # data.to_csv("GW_SS_data(2).csv", index = False)
    return data

def process_smoothness_structure_textures():
    data = []
    folder = os.getcwd() + "/data/Normalized_Brodatz/Sample/*"
    for file in glob.glob(folder):
        if len(data) == 20: 
            break
        img_id = file.split("/")[-1].split(".")[0]
        image_read = cv2.imread(file)
        # conversion numpy array into rgb image to show
        img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
        # Laplacian filter
        # img = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=5)).astype('float64')
        # Tiny random noise to avoid zero variance 
        ord_dis = ordpy.weighted_smoothness_structure(img, q = 0.25)
        data.append([img_id, ord_dis[0], ord_dis[1]])

    data = pd.DataFrame(data, columns = ["Label", "Smoothness \tau", "Curve structure \kappa"]).sort_values('Label').reset_index(drop = True)
    # data.to_csv("GW_SS_data(2).csv", index = False)
    return data

# data_hc, data_fs = process_data()

data_ss = process_smoothness_structure()
# data_ss = process_smoothness_structure_textures()

# data_ss = pd.read_csv("data/GW_SS_data(2).csv")
# data_hc = pd.read_csv("HC_data.csv")
# data_fs = pd.read_csv("FS_data.csv")

plt.rcParams.update({"font.size": 17})
# sns.jointplot(data = data_hc, x = 'Entropy', y = "Complexity", hue = 'Label', s = 70)
# plt.show()
# sns.jointplot(data = data_fs, x = 'Entropy', y = "Fisher information", hue = 'Label', s = 70)
# plt.show()
sns.jointplot(data = data_ss, x = 'Smoothness', y = "Curve structure", hue = 'Label', s = 70)
# sns.jointplot(data = data_ss, x = '1', y = "2", hue = '3', s = 70)
plt.show()