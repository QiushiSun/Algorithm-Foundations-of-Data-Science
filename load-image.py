
# alpha一般取0.9，case study的时候可以多取几个

# 调试过程中打印参数，如特征向量个数

# 去中心化后，还原的时候要把中心加回来，才会是原来的图像

import imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

a=imageio.imread("beach/beach00.tif")


a_np=np.array(a)
a_r=a_np[:,:,0] #R
a_g=a_np[:,:,1] #G
a_b=a_np[:,:,2] #B

def loadImage(path):

    img = Image.open(path)
    path=""
    img = img.convert("L")
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    data = np.array(data).reshape(height,width)/100
    new_im = Image.fromarray(data*100)
    new_im.show()

    return data

loadImage("airplane/airplane00.tif")




