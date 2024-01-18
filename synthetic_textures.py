import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import ordpy
import pandas as pd
import seaborn as sns

def smooth_noise(N, k):
    np.random.seed(N)
    noise = np.uint8(255 * np.random.rand(N, N))
    count = 2
    while(k >= count):
        noise = ndimage.zoom(noise, 2)[:N, :N]
        count *= 2
    # print(noise.shape)
    # plt.imshow(noise, cmap='gray')
    # plt.show()
    return noise

def turbulence(N, k):
    np.random.seed(N)
    noise = np.random.rand(N, N)
    turb = noise.copy()
    sum = 1 #
    count = 2
    while(k >= count):
        noise = ndimage.zoom(noise, 2)[:N, :N]
        turb = turb + count * noise
        sum += count #
        count *= 2
    # turb = np.uint8(128 * turb / k)
    turb = np.uint8(255 * turb / sum) #
    # print(turb.shape)
    # plt.imshow(turb, cmap='gray')
    # plt.show()
    return turb

def marble(N, Wx, Wy, turb_power, k):
    x = Wx * np.linspace(0, 1, N).reshape(1,-1)
    y = Wy * np.linspace(0, 1, N).reshape(-1,1)
    turb = turb_power * turbulence(N, k).astype(float) / 256.0
    arg = np.tile(x, (N,1)) + np.tile(y, (1,N)) + turb
    sine2D = 128.0 + (127.0 * np.sin(np.pi * arg))
    sine2D = np.uint8(sine2D)
    print(sine2D.shape)
    plt.imshow(sine2D, cmap='gray')
    plt.show()
    return sine2D

def vortices(N, Wx, Wy, turb_power, k):
    x = Wx * np.linspace(0, 1, N).reshape(1,-1)
    y = Wy * np.linspace(0, 1, N).reshape(-1,1)
    turb = turb_power * smooth_noise(N, k).astype(float) / 256.0
    arg = np.tile(x, (N,1)) + np.tile(y, (1,N)) + turb
    sine2D = 128.0 + (127.0 * np.sin(np.pi * arg))
    sine2D = np.uint8(sine2D)
    return sine2D
    # sine2D = np.tile(sine1D, (N,1))
    print(sine2D.shape)
    plt.imshow(sine2D, cmap='gray')
    plt.show()

def wood(N, Wx, Wy, turb_power, k):
    x = np.power(np.linspace(-1, 1, N), 2).reshape(1,-1)
    y = np.power(np.linspace(-1, 1, N), 2).reshape(-1,1)
    turb = turb_power * turbulence(N, k).astype(float) / 256.0
    arg = np.sqrt(np.tile(x, (N,1)) + np.tile(y, (1,N))) + turb
    sine2D = 128.0 + (127.0 * np.sin(np.pi * Wx * Wy * arg))
    sine2D = np.uint8(sine2D)
    # sine2D = np.tile(sine1D, (N,1))
    print(sine2D.shape)
    plt.imshow(sine2D, cmap='gray')
    plt.show()
    return sine2D

def bees(N, Wx, Wy, turb_power, k):
    x = np.linspace(0, 1, N).reshape(1,-1)
    y = np.linspace(0, 1, N).reshape(-1,1)
    turb = turb_power * smooth_noise(N, k).astype(float) / 256.0
    arg_x = np.tile(x, (N,1)) + turb
    arg_y = np.tile(y, (1,N)) + turb
    sine2D = 255 * (np.sin(np.pi * Wx * arg_x) + np.sin(np.pi * Wy * arg_y) + 2) / 4
    sine2D = np.uint8(sine2D)
    # sine2D = np.tile(sine1D, (N,1))
    print(sine2D.shape)
    plt.imshow(sine2D, cmap='gray')
    plt.show()

def process_smoothness_structure(texture, N = 256, linspace = [0], Wx = 5, Wy = 5, q = 2):
    data = []
    for turb_power in linspace:
        k = 1
        while(k < N):
            # Generate a texture given its parameters
            if(texture == "smooth_noise"):
                img = smooth_noise(N, k)
            elif(texture == "turbulence"):
                img = turbulence(N, k)
            elif(texture == "vortices"):
                img = vortices(N, Wx = Wx, Wy = Wy, turb_power = turb_power, k = k)
            elif(texture == "wood"):
                img = wood(N, Wx = Wx, Wy = Wy, turb_power = turb_power, k = k)
            elif(texture == "marble"):
                img = marble(N, Wx = Wx, Wy = Wy, turb_power = turb_power, k = k)
            else:
                print("Unknown texture")
                return None
            # Compute Information Theory Metrics for images 
            if(q == 0): ord_dis = ordpy.smoothness_structure(img)
            else: ord_dis = ordpy.weighted_smoothness_structure(img, q)
            # Add to dataframe
            data.append([str(k), np.round(turb_power,2), ord_dis[0], ord_dis[1]])
            k *= 2
            # print(k)

    data = pd.DataFrame(data, columns = ["K", "TurbPower", "Smoothness", "Curve structure"])
    filename = f"{texture}_q={q}_N={N}_Wx={Wx}_Wy={Wy}_TP=({linspace[0], linspace[-1], len(linspace)})"
    data.to_csv(f"transformed_data/Synthetic/{filename}.csv", index = False)
    return data

# vortices(N = 256, Wx = 5, Wy = 10, turb_power = 10, k = 8)
# marble(N = 256, Wx = 5, Wy = 10, turb_power = 5, k = 32)
# wood(N = 256, Wx = 5, Wy = 5, turb_power = 0.1, k = 4)
# smooth_noise(N = 256, k = 32)
# turbulence(N = 256, k = 256)

# smooth_noise(N = 256, k = 2)
# smooth_noise(N = 256, k = 4)
# smooth_noise(N = 256, k = 8)
# smooth_noise(N = 256, k = 16)
# smooth_noise(N = 256, k = 32)
# smooth_noise(N = 256, k = 64)
# smooth_noise(N = 256, k = 128)

# 2 5 8

wood(N = 256, Wx = 5, Wy = 5, turb_power = 0.5, k = 2)
wood(N = 256, Wx = 5, Wy = 5, turb_power = 0.5, k = 8)
wood(N = 256, Wx = 5, Wy = 5, turb_power = 0.5, k = 32)
wood(N = 256, Wx = 5, Wy = 5, turb_power = 0.1, k = 128)

# # data_ss = process_smoothness_structure(texture = "vortices", linspace = np.linspace(0.1, 5, 50))
# data_ss = process_smoothness_structure(texture = "marble", linspace = np.linspace(2, 7, 50))#, Wx = 5, Wy = 10)
# # data_ss = process_smoothness_structure(texture = "wood", linspace = np.linspace(0.1, 5, 50))

# plt.rcParams.update({"font.size": 17})
# sns.lineplot(data = data_ss, x = 'Smoothness', y = "Curve structure", hue = 'K', style = 'K', markers = True, legend = None, zorder = 0)
# sns.scatterplot(data = data_ss, x = 'Smoothness', y = "Curve structure", hue = 'K', style = 'K', size = 'TurbPower', sizes = (20, 200), zorder = 10)
# plt.show()

# sns.lineplot(data = data_ss, x = 'Smoothness', y = "Curve structure", hue = 'TurbPower', markers = None, zorder = 0, palette = "coolwarm", legend = None)
# sns.scatterplot(data = data_ss, x = 'Smoothness', y = "Curve structure", hue = 'TurbPower', style = 'TurbPower', size = 'K', sizes = (20, 200), zorder = 10, palette = "coolwarm")
# plt.show()

