import numpy as np
import ordpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

'''Aqui nós geramos ruido colorido a partir do ruido branco uniforme. 
Calculos no dominio da frequencia (FD) segundo artigo de osvaldinho'''

def generate_colored_noise_2d_FD(shape, k):

    # ruido branco uniforme
    white_noise = np.random.uniform(-0.5, 0.5, shape)
    
    # Aplica a transformada e desloca as baixas frequencias para o centro
    y = np.fft.fftshift(np.fft.fft2(white_noise))

    ''' "matriz de frequência": 
    No caso 1-D, esta seria uma matriz das frequências reais que correspondem às amplitudes 
    dadas pela transformada. 
    No caso 2-D, esta é a distância do centro do nosso espaço de Fourier deslocado, 
    pois quanto mais longe vamos das bordas, maior será a frequência capturada naquele ponto'''
    _x, _y = np.mgrid[0:y.shape[0], 0:y.shape[1]]
    f = np.hypot(_x - y.shape[0] / 2, _y - y.shape[1] / 2)

    # ruido modificado
    y_2 = y / f**(k/2)

    colored_noise = np.nan_to_num(y_2, nan=0, posinf=0, neginf=0)


    # Retira o deslocamento e calcula a inversa da transformada 
    colored_noise = np.fft.ifft2(np.fft.ifftshift(colored_noise)).real

    # Normaliza o resultado
    # colored_noise /= np.std(colored_noise)

    return colored_noise

def process_smoothness_structure(shape, linspace = range(5), q = 2):
    data = []
    shape = (256, 256)
    for k in linspace:
        img = generate_colored_noise_2d_FD(shape, k)
        if(q == 0): ord_dis = ordpy.smoothness_structure(img)
        else: ord_dis = ordpy.weighted_smoothness_structure(img, q)
        data.append([k, ord_dis[0], ord_dis[1]])

    data = pd.DataFrame(data, columns = ["k", "Smoothness", "Curve structure"])
    # data.to_csv(f"transformed_data/Synthetic/{filename}.csv", index = False)
    return data

data_ss = process_smoothness_structure(shape = (256, 256), linspace = np.linspace(0, 6, num = 61), q = 0)

plt.rcParams.update({"font.size": 17})
sns.scatterplot(data = data_ss, x = 'Smoothness', y = "Curve structure", hue = 'k', palette = 'viridis', legend = None)
# plt.colorbar(cm.ScalarMappable(cmap=cm.viridis))
plt.ylim([-0.025, 0.5])
plt.xlim([-0.025, 0.7])
plt.show()