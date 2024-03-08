import numpy as np
import os
from PIL import Image
import sys
from mmseg.core.evaluation import mean_iou



def load_images(folder):
    """
    Load all images in the given folder. Returns a list of NumPy arrays.
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(folder, filename))
            images.append(np.array(img))
    return images



# Example usage
# true_label_folder = '/share/home/dq070/hy-tmp/AS_id_all/all-condition/all-img/gt/val/labelTrainIds'
# true_label_folder = '/share/home/dq070/hy-tmp/AS_id_all/fog-driving/gt2'
true_label_folder = '/share/home/dq070/lfh/fog_medium/fog/gt/val/labelTrainIds'
# true_label_folder='/share/home/dq070/hy-tmp/AS_id_all/bdd100k-night-87/gt/val/labelTrainIds'
# true_label_folder='/share/home/dq070/hy-tmp/AS_id_all/NighttimeDriving/gt/labelTrainIds'
# true_label_folder = '/share/home/dq070/hy-tmp/AS_id_all/NighttimeDriving/gt/labelTrainIds-rename'
pred_label_folder = '/share/home/dq070/CoT/Rein-Cityscapes/work_dirs/adapter_256_1024_8_8/fz/labelTrainIds'

true_labels = load_images(true_label_folder)
pred_labels = load_images(pred_label_folder)
# from mmseg.core.evaluation import mean_iou

# 假设你已经有了真实标签和预测标签
# true_labels 和 pred_labels 应该是相同维度的数组列表

# 计算IoU和mIoU
results = mean_iou(pred_labels, true_labels, num_classes=19, ignore_index=255)
sum = 0
for i in range(19):
    if results['IoU'][i] >0:
        sum += results['IoU'][i]
results_mIou = sum/19
# 输出结果
print(results)
print(results_mIou)