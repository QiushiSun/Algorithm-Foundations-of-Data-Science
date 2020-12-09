import numpy as np
import imageio
from PIL import Image
import os
import matplotlib.pyplot as plt

def comp_2d(image_2d,accuracy):
    cov_mat = image_2d - np.mean(image_2d,axis=0)
    eig_val,eig_vec = np.linalg.eig(np.cov(cov_mat))
    #特征值从大到小排序
    idx = np.argsort(eig_val)
    idx = idx[::-1]
    eig_vec = eig_vec[:,idx]
    eig_val = eig_val[idx]
    # 特征值个数选取
    eig_val_sum=np.sum(eig_val)
    p = np.linalg.matrix_rank(image_2d)  #rank
    part_eig_val = 0
    for i in range(p):
        part_eig_val+=eig_val[i]
        if(part_eig_val > 0.9*eig_val_sum):
            break
    numpc=i #选取主特征的个数

    eig_vec = eig_vec[:,range(numpc)]

#   特征值个数+eig_vec的大小+center

    #print(i)
    score = np.dot(eig_vec.T, cov_mat)
    recon = np.dot(eig_vec,score) + np.mean(image_2d,axis = 0).T
    recon_img_mat = np.uint8(np.absolute(recon))
    return recon_img_mat


def PCA_implement(image,accuracy):

    a_np = np.array(image)  # 旋转array
    a_r = a_np[:, :, 0]  # R
    a_g = a_np[:, :, 1]  # G
    a_b = a_np[:, :, 2]  # B
    a_r_recon, a_g_recon, a_b_recon = comp_2d(a_r,accuracy), comp_2d(a_g,accuracy), comp_2d(a_b,accuracy)
    recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon))
    recon_color_img = Image.fromarray(recon_color_img)
    return recon_color_img

# 怎么压缩的最小，图片质量又最高
# a=imageio.imread("airplane/airplane00.tif")

path = "beach"
files= os.listdir(path)
files.sort()
#print(files)
accuracy = input("please input accuracy:")
for file in files:
    print(file)
    img = imageio.imread('beach/'+str(file))
    processed_iamge=PCA_implement(img,accuracy)
    processed_iamge.save('new_beach/'+str(file))





