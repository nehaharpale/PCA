from PIL import Image
import matplotlib.pyplot as plt
import os
import pca
import numpy as np
import matplotlib.image as mpimgs

def load_data(input_dir):
    images =[]
    for file in os.listdir(input_dir):
            im = mpimgs.imread(input_dir+file)
            nim = im.flatten()
            images.append(nim)
    return np.array(images, np.float64).T

def compress_images(DATA,k):
    var = 0
    count = 0
    Z = pca.compute_Z(DATA, centering=True, scaling=False)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, var)
    PCS = PCS[:,: k]
    X_compressed = Z_star.dot(PCS.T)
    if not os.path.exists("./Output"):
         os.mkdir("./Output")
    for i in X_compressed.T:
        count += 1
        file_name = "./Output/"+str(count)+"compressed_k"+str(k)
        n = np.reshape(i,(60,48))
        plt.imsave(file_name,n,cmap='Greys_r')

#input_dir = "C:\\Users\\Neha\\Desktop\\ML\\Project 4\\Data\\Test\\"
#DATA = load_data(input_dir)
#k = 2000
#compress_images(DATA, k)
