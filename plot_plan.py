import matplotlib.pyplot as plt
import pandas as pd
import ordpy
import cv2
import glob
import os
import seaborn as sns

def process_data():
    data = []
    data_fisher = []
    folder = os.getcwd() + "/data/images/*.*"
    for file in glob.glob(folder):
        img_id = int(file.split("/")[-1].split(".")[0])
        image_read = cv2.imread(file)
        # conversion numpy array into rgb image to show
        img = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)       
        hc = ordpy.complexity_entropy(img, dx = 2, dy = 2, taux = 1, tauy = 1)
        data.append([img_id, hc[0], hc[1]])
        fs = ordpy.fisher_shannon(img, dx = 2, dy = 2, taux = 1, tauy = 1)
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
    # data.to_csv("HC_data.csv", index = False)
    # data_fisher.to_csv("FS_data.csv", index = False)
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
        ord_dis = ordpy.ordinal_distribution(img, dx = 2, dy = 2, taux = 1, tauy = 1)
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
    data.to_csv("BP_data.csv", index = False)
    return data

# data_hc, data_fs = process_data()

process_bandt_pomp()

data_hc = pd.read_csv("HC_data.csv")
data_fs = pd.read_csv("FS_data.csv")

plt.rcParams.update({"font.size": 17})
sns.jointplot(data_hc, x = 'Entropy', y = "Complexity", hue = 'Label', s = 70)
plt.show()
sns.jointplot(data_fs, x = 'Entropy', y = "Fisher information", hue = 'Label', s = 70)
plt.show()