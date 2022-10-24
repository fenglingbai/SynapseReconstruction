# connect 2D instance according to connection log
import os
import cv2
import pickle
import numpy as np
import joblib
from skimage import io
# connection log path
save_path = "constrution_outcome"
# fused inference data path
root_path = "fusion_outcome_v1"
data_list = os.listdir(root_path)
data_list.sort(key=lambda x: int(x[:-7]))
# load connection relationship
construction_path = os.path.join(save_path, "constrution_outcome.pickle")
with open(construction_path, 'rb') as f_in:
    construction_data = joblib.load(f_in)
    print('construction_data has been read from {}'.format(construction_path))

construction_record = construction_data["construction_record"]


item_value = 10000
label16_path = "label16_outcome"
raw_path = "raw_outcome"
os.makedirs(label16_path, exist_ok=True)
os.makedirs(raw_path, exist_ok=True)
# 保存tif stack
label_stack = np.zeros(shape=(0, 2048, 2048))

for i in range(len(data_list)):
    data_item = data_list[i]
    data_item = os.path.join(root_path, data_item)
    with open(data_item, 'rb') as f_in:
        data = joblib.load(f_in)
        print('data has been read from {}'.format(data_item))
    # save label
    pre_mask_data = data['pre_mask']
    pre_mask_data = pre_mask_data.astype(np.uint16) * item_value
    add_results = np.zeros(shape=(pre_mask_data.shape[0], pre_mask_data.shape[1]), dtype=np.uint16)
    add_layer_index = np.where(construction_record[i] != -1)[0]
    # add layers
    for j in add_layer_index:
        add_results = add_results + pre_mask_data[:, :, j] * construction_record[i, j]
    # save data
    label_stack = np.concatenate([label_stack, add_results[np.newaxis, :, :]], axis=0)
    # # 保存png
    # cv2.imwrite(os.path.join(label16_path, '%04d.png' % i), add_results, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # print(os.path.join(label16_path, '%04d.png' % i)+" has already saved")
    # # save image
    # image_data = data['img'][:, :, 0]
    # cv2.imwrite(os.path.join(raw_path, '%04d.png' % i), image_data, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    # print(os.path.join(raw_path, '%04d.png' % i) + " has already saved")
# save tif stack
io.imsave('label16_v4.tif', label_stack.astype(np.uint16))
print("end")