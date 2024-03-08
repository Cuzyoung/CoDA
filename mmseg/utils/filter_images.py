import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import sys
import tqdm
# 定义路径
source_path = '/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit/train-all'
dark_path = '/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit//severe_images'
not_dark_path = '/share/home/dq070/hy-tmp/datasets/cityscapes/leftImg8bit/not_severe_images'

# 创建目标文件夹，如果它们不存在
Path(dark_path).mkdir(parents=True, exist_ok=True)
Path(not_dark_path).mkdir(parents=True, exist_ok=True)

dark_threshold = 40 
dark_prop_threshold = 0.18
n =0
# 遍历源路径中的所有图片文件
for img_file in os.listdir(source_path):
    img_path = os.path.join(source_path, img_file)
    if os.path.isfile(img_path):
        # 读取图片并转换为灰度图
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(gray_img)
        # sys.exit()
        # 计算暗部比例
        dark_sum = np.sum(gray_img < dark_threshold)
        dark_prop = dark_sum / gray_img.size
        
        # 根据暗部比例移动文件
        if dark_prop >= dark_prop_threshold:
            # n += 1
            
            shutil.copy(img_path, os.path.join(dark_path, img_file))
        else:
            shutil.copy(img_path, os.path.join(not_dark_path, img_file))
            pass
print(n)