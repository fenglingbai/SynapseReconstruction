# 存储版本

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
import joblib
import matplotlib.pyplot as plt


# 根目录
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
########################################################################################################################
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import Config
import utils
import model as modellib
import visualize
from model import log

# Save logs and trained model path
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

DATA_DIR = "your/root/path"
# image
RAW_DIR = os.path.join(DATA_DIR, "raw")
# gt label
LABEL_DIR = os.path.join(DATA_DIR, "label16")


# os.path.exists(COCO_MODEL_PATH)
# ## Configurations
########################################################################################################################



class SynapsesConfig(Config):
    """
    Synapse Config Setting
    """
    # Task name
    NAME = "synapses"
    # 在一块gpu上训练，每块GPU训练一张图片，Batch size 为1 (GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # 类别数量 (包含背景)
    NUM_CLASSES = 1 + 1  # background + synapse

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # 将电镜图片的灰度图转化为彩图，所以选择3
    IMAGE_CHANNEL_COUNT = 3

    # anchor像素占的边长
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # 每张图片挑取得分最高的TRAIN_ROIS_PER_IMAGE个roi区域进行训练
    # 最好的效果是推荐的roi区域有33%的正样本ROIs.
    TRAIN_ROIS_PER_IMAGE = 16

    # 每一代训练100个steps
    STEPS_PER_EPOCH = 300

    # 验证会进行在当前epoch结束后进行
    # validation_steps设置了验证使用的validation data steps数量(batch数量)
    # 如validation batch size(没必要和train batch相等)=64
    # validation_steps=100,(steps相当于batch数, 应<=TotalvalidationSamples / ValidationBatchSize)
    # 则会从validation data中取6400个数据用于验证
    # (如果一次step后validation data set剩下的data足够下一次step,会继续从剩下的data set中选取, 如果不够会重新循环).
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 144


    # learning rate will be reduced after this many epochs if the validation loss is not improving
    # 经过"patience"迭代次数之后，验证损失没有得到改善，则学习率将会降低
    LEARNING_RATE_PATIENCE = 10
    # 降低学习率的倍数
    LEARNING_RATE_DROP = 0.5

    # training will be stopped after this many epochs without the validation loss improving
    # 经过"early_stop"迭代次数之后，验证损失没有得到改善，采取早停策略
    EARLY_STOPPING_PATIENCE = 50

    # 保存loss信息
    LOGGING_FILE = "training.log"

# 初始化参数配置
config = SynapsesConfig()
# 展示当前参数
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    # _ 是图像对象，ax 是坐标轴对象
    # rows行cols列(size * cols)x(size * rows)大小的子图
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class SynapsesDataset(utils.Dataset):

    def load_shapes(self, raw_path, label_path, start, count, slice, height, width, type):
        # 增加数据集的类别信息与图片信息
        self.add_class("EMdata", 1, "synapses")
        # 完成后的形式
        # self.class_info = [{"source": "", "id": 0, "name": "BG"},
        #                    {"source": "EMdata", "id": 1, "name": "synapses"}]
        raw_path_list = os.listdir(raw_path)
        label_path_list = os.listdir(label_path)
        assert len(raw_path_list) == len(label_path_list), "raw list must equal to label list."
        raw_path_list.sort(key=lambda x: int(x[:-4]))
        label_path_list.sort(key=lambda x: int(x[:-4]))
        # 添加图片
        for i in range(count):
            for j in range(slice + 1):
                for k in range(slice + 1):
                    # 将该图片信息录入到image_info中
                    # 完成后的形式为self.image_info =
                    # [{"id": 0,"source":"shapes","path":None,...},
                    #  {"id": 1,"source":"shapes","path":None,...},
                    #  ...
                    #  {"id": i,"source":"shapes","path":None,...}]
                    self.add_image("EMdata", image_id=i * (slice + 1) * (slice + 1) + j * (slice + 1) + k, path=None,
                                   path_to_raw=os.path.join(raw_path, raw_path_list[i + start]),
                                   path_to_label=os.path.join(label_path, label_path_list[i + start]),
                                   position_height=j, position_width=k,
                                   width=width, height=height, slice=slice, type=type)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        # 加载当前数据集中标号为image_id的图片信息
        # 例如：
        # {"id": 0,"source":"shapes","path":None,"width"=width,"height"=height,"bg_color"=bg_color,"shapes"=shapes}
        # 其中shapes包含所有检测对象的shape，color与dims
        info = self.image_info[image_id]
        height = info['height']
        width = info['width']
        slice = info['slice']
        j = info['position_height']
        k = info['position_width']
        img = cv2.imread(info['path_to_raw'], -1)
        img = cv2.resize(img, dsize=(height * slice, width * slice), interpolation=cv2.INTER_CUBIC)
        out_img = img[int(j * height * ((slice - 1) / slice)):int(j * height * ((slice - 1) / slice) + height),
                  int(k * width * ((slice - 1) / slice)):int(k * width * ((slice - 1) / slice) + width)]
        out_img = out_img[:, :, np.newaxis]
        out_img = np.concatenate((out_img, out_img, out_img), axis=2)
        return out_img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        elif info["source"] == "synapses":
            return info["path_to_raw"], info["position_height"], info["position_width"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        # 加载当前数据集中标号为image_id的图片信息
        # 例如：
        # {"id": 0,"source":"shapes","path":None,"width":width,"height":height,"bg_color":bg_color,"shapes":shapes}
        # {'id': 12, 'source': 'shapes', 'path': None, 'width': 128, 'height': 128, 'bg_color': array([193, 183,  40]), 'shapes': [('triangle', (2, 89, 218), (100, 93, 30)), ('circle', (156, 175, 70), (101, 62, 27))]}
        info = self.image_info[image_id]
        height = info['height']
        width = info['width']
        slice = info['slice']
        j = info['position_height']
        k = info['position_width']
        label = cv2.imread(info['path_to_label'], -1)
        label = label.astype(np.uint8)
        label = cv2.resize(label, dsize=(height * slice, width * slice), interpolation=cv2.INTER_CUBIC)
        label[label != 0] = 255
        # 将子区域从原图中裁剪出来
        temp_label = label[int(j * height * ((slice - 1) / slice)):int(j * height * ((slice - 1) / slice) + height),
                     int(k * width * ((slice - 1) / slice)):int(k * width * ((slice - 1) / slice) + width)]

        # 寻找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp_label, connectivity=8,
                                                                                ltype=None)
        # 取出当前所有连通域的面积
        out_area = stats[:, 4]
        # 根据连通域的大小进行对象的筛选
        out_sort = np.where((out_area > (2000 // 16)) & (out_area < (100000 // 16)))[0]
        # 取出剩余的连通域面积
        out_area = stats[out_sort, 4]
        # 根据连通域面积的从大到小取出对应的索引
        out_sort = out_sort[np.argsort(out_area)][::-1]
        count = len(out_sort)
        mask = np.zeros([count, info['height'], info['width']], dtype=np.uint8)
        for item in range(count):
            single_label = labels.copy().astype(np.uint8)
            single_label[single_label != out_sort[item]] = 0
            single_label[single_label == out_sort[item]] = 255
            mask[item] = single_label
        mask = np.transpose(mask, [1, 2, 0])
        if count > 0:
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                # 从count - 2倒序至0
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        # 找到检测目标所属的id编号，全都是突触，所以构造全1数组
        class_ids = np.ones((count,), dtype=np.int)
        return mask.astype(np.bool), class_ids.astype(np.int32)


# Training dataset
dataset_train = SynapsesDataset()
dataset_train.load_shapes(RAW_DIR, LABEL_DIR, start=0, count=80, slice=4,
                          height=config.IMAGE_SHAPE[0], width=config.IMAGE_SHAPE[1], type="train")
dataset_train.prepare()

# Validation dataset
dataset_val = SynapsesDataset()
dataset_val.load_shapes(RAW_DIR, LABEL_DIR, start=80, count=9, slice=4,
                          height=config.IMAGE_SHAPE[0], width=config.IMAGE_SHAPE[1], type="val")
dataset_val.prepare()

# Validation dataset
dataset_test = SynapsesDataset()
dataset_test.load_shapes(RAW_DIR, LABEL_DIR, start=89, count=89, slice=4,
                          height=config.IMAGE_SHAPE[0], width=config.IMAGE_SHAPE[1], type="test")
dataset_test.prepare()

# Load and display random samples
# 在所有的训练集的image_ids中找1个对象
image_ids = np.random.choice(dataset_train.image_ids, 1)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=2)


class InferenceConfig(SynapsesConfig):
    # 推理参数设置，这里设置为一张一张推理
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.95

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

def single_fusion(rois, masks, class_ids, scores, threshold=0.1):
    height = rois[:, 2] - rois[:, 0]
    width = rois[:, 3] - rois[:, 1]
    center_y = rois[:, 0] + 0.5 * height
    center_x = rois[:, 1] + 0.5 * width

    instance_num = rois.shape[0]
    # 融合索引记录：-1为已融合需要保留的instance，other为融进other的instance，需要后续删除
    fusion_id = np.ones(shape=instance_num, dtype=int) * -1

    for i_item in range(instance_num - 1):
        for j_item in range(i_item + 1, instance_num):
            if 2 * abs(center_y[i_item] - center_y[j_item]) < height[i_item] + height[j_item] and \
                    2 * abs(center_x[i_item] - center_x[j_item]) < width[i_item] + width[j_item]:
                # 检查mask
                mask_y1 = min(rois[i_item, 0], rois[j_item, 0])
                mask_y2 = max(rois[i_item, 2], rois[j_item, 2])
                mask_x1 = min(rois[i_item, 1], rois[j_item, 1])
                mask_x2 = max(rois[i_item, 3], rois[j_item, 3])
                # 提取掩膜
                masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, i_item], (-1, 1)).astype(
                    np.float32)
                masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j_item], (-1, 1)).astype(
                    np.float32)
                # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                area1 = np.sum(masks1, axis=0)[0]
                area2 = np.sum(masks2, axis=0)[0]

                # intersections and union
                # 计算点积,即交集,得到结果(instances1, instances2)
                intersections = np.dot(masks1.T, masks2)[0][0]
                # 计算并集
                # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                union = area1 + area2 - intersections
                # 得到结果(instances1, instances2)的iou矩阵
                overlaps = intersections / union
                # 包围盒有重合，计算maskiou
                if overlaps > threshold:
                    # j_item融入i_item中
                    index_fusion = i_item
                    index_delet = j_item
                    while fusion_id[index_fusion] != -1:
                        assert fusion_id[
                                   index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                        index_fusion = fusion_id[index_fusion]
                    if fusion_id[index_delet] == index_fusion:
                        # 前面的操作已经融合过，可以省略
                        pass
                    else:
                        # 这两个对象前操作没有融合过，进行融合
                        # 需要融合的instance为需要融合的最小索引，可以直接融合
                        rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                        rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                        rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                        rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                        masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                        assert class_ids[index_fusion] == class_ids[
                            index_delet], 'you need to change this code to match the Multiclass classification'
                        scores[index_fusion] = max(scores[index_fusion], scores[index_delet])
                        fusion_id[index_delet] = index_fusion

    delet = np.where(fusion_id != -1)[0]
    rois = np.delete(rois, delet, axis=0)
    masks = np.delete(masks, delet, axis=2)
    class_ids = np.delete(class_ids, delet, axis=0)
    scores = np.delete(scores, delet, axis=0)
    return rois, masks, class_ids, scores

def aug_fusion(rois, masks, class_ids, scores, aug_num=4, aug_target=0.5, threshold=0.1):
    height = rois[:, 2] - rois[:, 0]
    width = rois[:, 3] - rois[:, 1]
    center_y = rois[:, 0] + 0.5 * height
    center_x = rois[:, 1] + 0.5 * width

    # 初始化需要保留的mask计数器
    mask_num = 0
    # 记录synapse的字典
    mask_item = {}

    instance_num = rois.shape[0]
    # 融合索引记录：-1为已融合需要保留的instance，other为融进other的instance，需要后续删除
    fusion_id = np.ones(shape=instance_num, dtype=int) * -1
    fusion_count = np.ones(shape=instance_num, dtype=int)
    instance_type = np.linspace(0, instance_num, num=instance_num, endpoint=False, dtype=np.int)
    for i_item in range(instance_num - 1):
        for j_item in range(i_item + 1, instance_num):
            if 2 * abs(center_y[i_item] - center_y[j_item]) < height[i_item] + height[j_item] and \
                    2 * abs(center_x[i_item] - center_x[j_item]) < width[i_item] + width[j_item]:
                # 检查mask
                mask_y1 = min(rois[i_item, 0], rois[j_item, 0])
                mask_y2 = max(rois[i_item, 2], rois[j_item, 2])
                mask_x1 = min(rois[i_item, 1], rois[j_item, 1])
                mask_x2 = max(rois[i_item, 3], rois[j_item, 3])
                # 提取掩膜
                masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, i_item], (-1, 1)).astype(
                    np.float32)
                masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j_item], (-1, 1)).astype(
                    np.float32)
                # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                area1 = np.sum(masks1, axis=0)[0]
                area2 = np.sum(masks2, axis=0)[0]

                # intersections and union
                # 计算点积,即交集,得到结果(instances1, instances2)
                intersections = np.dot(masks1.T, masks2)[0][0]
                # 计算并集
                # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                union = area1 + area2 - intersections
                # 得到结果(instances1, instances2)的iou矩阵
                overlaps = intersections / union
                # 包围盒有重合，计算maskiou
                if overlaps > threshold:
                    # 因为两平行的斜的突触包围盒也会有重叠，所以需要有Mask判断
                    # j_item融入i_item中聚类
                    instance_type[instance_type == j_item] = instance_type[i_item]

                    # ################################
                    # index_fusion = i_item
                    # index_delet = j_item
                    # while fusion_id[index_fusion] != -1:
                    #     assert fusion_id[
                    #                index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                    #     index_fusion = fusion_id[index_fusion]
                    # if fusion_id[index_delet] == index_fusion:
                    #     # 前面的操作已经融合过，可以省略
                    #     pass
                    # else:
                    #     # 这两个对象,前操作没有融合过，进行融合
                    #     # 需要融合的instance为需要融合的最小索引，可以直接融合
                    #     rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                    #     rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                    #     rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                    #     rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                    #     masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                    #     assert class_ids[index_fusion] == class_ids[
                    #         index_delet], 'you need to change this code to match the Multiclass classification'
                    #     scores[index_fusion] = scores[index_fusion] + scores[index_delet]
                    #     fusion_id[index_delet] = index_fusion
                    #     fusion_count[index_fusion] = fusion_count[index_fusion] + 1

    instance_only_type = np.unique(instance_type)
    out_rois = np.zeros(shape=(0, 4), dtype=np.int32)
    out_mask = np.zeros(shape=(masks.shape[0], masks.shape[1], 0), dtype=np.float32)
    out_class_ids = np.zeros(shape=0, dtype=np.int32)
    out_scores = np.zeros(shape=0, dtype=np.float32)

    for cluster_type in instance_only_type:
        # 找出所有属于这一类的实例坐标
        cluster_indexs = np.where(instance_type == cluster_type)[0]
        assert cluster_indexs.shape[0] > 0, 'must have instance'
        # 开始叠加mask
        sinle_count = 0
        single_scores = 0.0
        single_mask = np.zeros(shape=(masks.shape[0], masks.shape[1]), dtype=np.float32)
        single_mask_bool = np.zeros(shape=(masks.shape[0], masks.shape[1]), dtype=np.bool)
        single_class_ids = class_ids[cluster_indexs[0]]
        for single_indexs in cluster_indexs:
            assert class_ids[single_indexs] == single_class_ids
            single_mask = single_mask + masks[:, :, single_indexs] * scores[single_indexs]
            single_scores = single_scores + scores[single_indexs]
            sinle_count = sinle_count + 1
        threshold = aug_num * aug_target
        single_mask_bool[single_mask > threshold] = True
        if np.sum(single_mask_bool) == 0:
            pass
        else:
            # 存在instance
            # 计算bboxes
            # Bounding box.
            horizontal_indicies = np.where(np.any(single_mask_bool, axis=0))[0]
            vertical_indicies = np.where(np.any(single_mask_bool, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            single_rois = np.array([y1, x1, y2, x2])
            single_scores = single_scores / sinle_count
            # 加入输出数据中
            out_rois = np.concatenate((out_rois, single_rois[np.newaxis, :]), axis=0)
            out_mask = np.concatenate((out_mask, single_mask_bool[:, :, np.newaxis]), axis=2)
            out_class_ids = np.concatenate((out_class_ids, np.array([single_class_ids])), axis=0)
            out_scores = np.concatenate((out_scores, np.array([single_scores])), axis=0)

    return out_rois, out_mask, out_class_ids, out_scores


def aug_only_fusion(rois, masks, class_ids, scores, aug_num=4, aug_target=0.5, threshold=0.1):
    height = rois[:, 2] - rois[:, 0]
    width = rois[:, 3] - rois[:, 1]
    center_y = rois[:, 0] + 0.5 * height
    center_x = rois[:, 1] + 0.5 * width

    # 初始化需要保留的mask计数器
    mask_num = 0
    # 记录synapse的字典
    mask_item = {}

    instance_num = rois.shape[0]
    # 融合索引记录：-1为已融合需要保留的instance，other为融进other的instance，需要后续删除
    fusion_id = np.ones(shape=instance_num, dtype=int) * -1
    fusion_count = np.ones(shape=instance_num, dtype=int)
    instance_type = np.linspace(0, instance_num, num=instance_num, endpoint=False, dtype=np.int)
    for i_item in range(instance_num - 1):
        for j_item in range(i_item + 1, instance_num):
            if 2 * abs(center_y[i_item] - center_y[j_item]) < height[i_item] + height[j_item] and \
                    2 * abs(center_x[i_item] - center_x[j_item]) < width[i_item] + width[j_item]:
                # 检查mask
                mask_y1 = min(rois[i_item, 0], rois[j_item, 0])
                mask_y2 = max(rois[i_item, 2], rois[j_item, 2])
                mask_x1 = min(rois[i_item, 1], rois[j_item, 1])
                mask_x2 = max(rois[i_item, 3], rois[j_item, 3])
                # 提取掩膜
                masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, i_item], (-1, 1)).astype(
                    np.float32)
                masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j_item], (-1, 1)).astype(
                    np.float32)
                # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                area1 = np.sum(masks1, axis=0)[0]
                area2 = np.sum(masks2, axis=0)[0]

                # intersections and union
                # 计算点积,即交集,得到结果(instances1, instances2)
                intersections = np.dot(masks1.T, masks2)[0][0]
                # 计算并集
                # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                union = area1 + area2 - intersections
                # 得到结果(instances1, instances2)的iou矩阵
                overlaps = intersections / union
                # 包围盒有重合，计算maskiou
                if overlaps > threshold:
                    # j_item融入i_item中
                    index_fusion = i_item
                    index_delet = j_item
                    while fusion_id[index_fusion] != -1:
                        assert fusion_id[
                                   index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                        index_fusion = fusion_id[index_fusion]
                    if fusion_id[index_delet] == index_fusion:
                        # 前面的操作已经融合过，可以省略
                        pass
                    else:
                        # 这两个对象前操作没有融合过，进行融合
                        # 需要融合的instance为需要融合的最小索引，可以直接融合
                        rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                        rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                        rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                        rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                        masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                        assert class_ids[index_fusion] == class_ids[
                            index_delet], 'you need to change this code to match the Multiclass classification'
                        scores[index_fusion] = scores[index_fusion] + scores[index_delet]
                        fusion_id[index_delet] = index_fusion
                        fusion_count[index_fusion] = fusion_count[index_fusion] + 1

    delet = np.where(fusion_count / aug_num < aug_target)[0]
    rois = np.delete(rois, delet, axis=0)
    masks = np.delete(masks, delet, axis=2)
    class_ids = np.delete(class_ids, delet, axis=0)
    fusion_count = np.delete(fusion_count, delet, axis=0)
    scores = np.delete(scores, delet, axis=0) / fusion_count
    return rois, masks, class_ids, scores


def all_fusion(rois, masks, class_ids, scores, slice_id, threshold=0.1, height_length=5, width_length=5):

    #             save_data["pre_bbox"] = detect_temp_bbox
    #             save_data["pre_mask"] = detect_temp_mask
    #             save_data["pre_class_id"] = detect_temp_class_id
    #             save_data["pre_score"] = detect_temp_scores
    #             save_data["pre_slice_id"] = detect_temp_slice_id

    height = rois[:, 2] - rois[:, 0]
    width = rois[:, 3] - rois[:, 1]
    center_y = rois[:, 0] + 0.5 * height
    center_x = rois[:, 1] + 0.5 * width

    instance_num = rois.shape[0]
    # 融合索引记录：-1为已融合需要保留的instance，other为融进other的instance，需要后续删除
    fusion_id = np.ones(shape=instance_num, dtype=int) * -1
    fusion_count = np.ones(shape=instance_num, dtype=int)
    slice_index = []
    # 遍历所有的区域，找出区域的起始突触的Index
    for i in range(height_length * width_length):
        if len(np.where(slice_id == i)[0]) == 0:
            # 该区域没有突触实例
            if len(slice_index) != 0:
                # 分块列表存在数据
                # slice_index.append(slice_index[-1])
                slice_index.append(-1)
            else:
                # 分块列表不存在数据，一直填充0
                slice_index.append(0)
        else:
            slice_index.append(np.where(slice_id == i)[0][0])
    # 加入中止标志
    slice_index.append(instance_num)
    # 由于之前的分块id标记中，若分块中没有数据，用-1表示，这里需要对slice_index中的-1进行处理
    # 反向遍历slice_index，如果有-1的地方，-1变为其后面的非-1的数字
    for slice_index_adjuet in range(len(slice_index)):
        if slice_index[-1-slice_index_adjuet] == -1:
            slice_index[-1 - slice_index_adjuet] = slice_index[-slice_index_adjuet]
    assert instance_num == masks.shape[2] == class_ids.shape[0] == scores.shape[0] == \
           slice_id.shape[0], 'instance number must be equal ! '

    for i in range(height_length * width_length):
        # 遍历不同的子图，进行局部融合
        # 块内融合的索引序列
        compute_self_index = [index for index in range(slice_index[i], slice_index[i+1])]


        # 块间融合的索引序列
        if (i + 1) % width_length != 0 and i < height_length * width_length - width_length:
            # 当前块不为最后一列,且不为最后一行
            compute_other_index = [index for index in range(slice_index[i + 1], slice_index[i + 2])] + [index for index in range(
                slice_index[i + width_length], slice_index[i + width_length + 2])]
        elif (i + 1) % width_length != 0:
            # 当前块不为最后一列,但为最后一行
            compute_other_index = [index for index in range(slice_index[i + 1], slice_index[i + 2])]
        elif i < height_length * width_length - width_length:
            # 当前块为最后一列,但不为最后一行
            compute_other_index = [index for index in range(
                slice_index[i + width_length], slice_index[i + width_length + 1])]
        else:
            compute_other_index = []


        for j in compute_self_index:
            for k in compute_self_index:
                if j < k:
                    # 进行块间融合
                    if 2 * abs(center_y[j] - center_y[k]) < height[j] + height[k] and 2 * abs(center_x[j] - center_x[k]) < \
                            width[j] + width[k]:
                        # 找到最大的非背景区域
                        mask_y1 = min(rois[j, 0], rois[k, 0])
                        mask_y2 = max(rois[j, 2], rois[k, 2])
                        mask_x1 = min(rois[j, 1], rois[k, 1])
                        mask_x2 = max(rois[j, 3], rois[k, 3])
                        # 提取掩膜
                        masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j] > .5, (-1, 1)).astype(
                            np.float32)
                        masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, k] > .5, (-1, 1)).astype(
                            np.float32)
                        # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                        area1 = np.sum(masks1, axis=0)[0]
                        area2 = np.sum(masks2, axis=0)[0]

                        # intersections and union
                        # 计算点积,即交集,得到结果(instances1, instances2)
                        intersections = np.dot(masks1.T, masks2)[0][0]
                        # 计算并集
                        # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                        union = area1 + area2 - intersections
                        # 得到结果(instances1, instances2)的iou矩阵
                        overlaps = intersections / union
                        # 包围盒有重合，计算maskiou
                        if overlaps > threshold:
                            # k融入j中,可能出现的情况：A,B无交集，C为AC的并集，按A,B,C的aug_fusion，
                            # 可能出现C,C,delet的情况，这里进行合并删除
                            index_fusion = j
                            index_delet = k
                            while fusion_id[index_fusion] != -1:
                                assert fusion_id[
                                           index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                                index_fusion = fusion_id[index_fusion]
                            if fusion_id[index_delet] == index_fusion:
                                # 前面的操作已经融合过，可以省略
                                pass
                            else:
                                # 需要融合的instance为需要融合的最小索引，可以直接融合
                                rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                                rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                                rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                                rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                                masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                                assert class_ids[index_fusion] == class_ids[
                                    index_delet], 'you need to change this code to match the Multiclass classification'
                                scores[index_fusion] = scores[index_fusion] + scores[index_delet]
                                fusion_id[index_delet] = index_fusion
                                fusion_count[index_fusion] = fusion_count[index_fusion] + 1
            # 块间融合
            for k in compute_other_index:
                if 2 * abs(center_y[j] - center_y[k]) < height[j] + height[k] and 2 * abs(center_x[j] - center_x[k]) < \
                        width[j] + width[k]:
                    # 找到最大的非背景区域
                    mask_y1 = min(rois[j, 0], rois[k, 0])
                    mask_y2 = max(rois[j, 2], rois[k, 2])
                    mask_x1 = min(rois[j, 1], rois[k, 1])
                    mask_x2 = max(rois[j, 3], rois[k, 3])
                    # 提取掩膜
                    masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j] > .5, (-1, 1)).astype(
                        np.float32)
                    masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, k] > .5, (-1, 1)).astype(
                        np.float32)
                    # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                    area1 = np.sum(masks1, axis=0)[0]
                    area2 = np.sum(masks2, axis=0)[0]

                    # intersections and union
                    # 计算点积,即交集,得到结果(instances1, instances2)
                    intersections = np.dot(masks1.T, masks2)[0][0]
                    # 计算并集
                    # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                    union = area1 + area2 - intersections
                    # 得到结果(instances1, instances2)的iou矩阵
                    overlaps = intersections / union
                    # 包围盒有重合，计算maskiou
                    if overlaps > threshold:
                        # k融入j中
                        index_fusion = j
                        index_delet = k
                        while fusion_id[index_fusion] != -1:
                            assert fusion_id[
                                       index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                            index_fusion = fusion_id[index_fusion]
                        if fusion_id[index_delet] == index_fusion:
                            # 前面的操作已经融合过，可以省略
                            pass
                        else:
                            # 需要融合的instance为需要融合的最小索引，可以直接融合
                            rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                            rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                            rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                            rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                            masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                            assert class_ids[index_fusion] == class_ids[
                                index_delet], 'you need to change this code to match the Multiclass classification'
                            scores[index_fusion] = scores[index_fusion] + scores[index_delet]
                            fusion_id[index_delet] = index_fusion
                            fusion_count[index_fusion] = fusion_count[index_fusion] + 1
    # delet
    delet = np.where(fusion_id != -1)[0]
    detect_bbox = np.delete(rois, delet, axis=0)
    detect_mask = np.delete(masks, delet, axis=2)
    detect_class_id = np.delete(class_ids, delet, axis=0)
    detect_scores = np.delete(scores, delet, axis=0)
    detect_slice_id = np.delete(slice_id, delet, axis=0)
    fusion_count = np.delete(fusion_count, delet, axis=0)

    # handle overlap
    delet_overlap = np.zeros(shape=detect_bbox.shape[0], dtype=int)
    # 0:不删除；1：删除
    for overlap_index in range(detect_bbox.shape[0]):
        row_in_overlap = (detect_bbox[overlap_index, 0] > 384 and detect_bbox[overlap_index, 2] < 512) or \
                         (detect_bbox[overlap_index, 0] > 768 and detect_bbox[overlap_index, 2] < 896) or \
                         (detect_bbox[overlap_index, 0] > 1152 and detect_bbox[overlap_index, 2] < 1280) or \
                         (detect_bbox[overlap_index, 0] > 1536 and detect_bbox[overlap_index, 2] < 1664)

        col_in_overlap = (detect_bbox[overlap_index, 1] > 384 and detect_bbox[overlap_index, 3] < 512) or \
                         (detect_bbox[overlap_index, 1] > 768 and detect_bbox[overlap_index, 3] < 896) or \
                         (detect_bbox[overlap_index, 1] > 1152 and detect_bbox[overlap_index, 3] < 1280) or \
                         (detect_bbox[overlap_index, 1] > 1536 and detect_bbox[overlap_index, 3] < 1664)

        """
            delet = np.where(fusion_count / aug_num < aug_target)[0]
    rois = np.delete(rois, delet, axis=0)
    masks = np.delete(masks, delet, axis=2)
    class_ids = np.delete(class_ids, delet, axis=0)
    fusion_count = np.delete(fusion_count, delet, axis=0)
    scores = np.delete(scores, delet, axis=0) / fusion_count
        """
        if row_in_overlap and col_in_overlap:
            # 四重叠区域
            if fusion_count[overlap_index] / 4 < 0.5:
                # 需要删除
                delet_overlap[overlap_index] = 1
        elif row_in_overlap or col_in_overlap:
            # 二重叠区域
            if fusion_count[overlap_index] / 2 < 0.5:
                # 需要删除
                delet_overlap[overlap_index] = 1

    # delet overlap instance
    delet_overlap_index = np.where(delet_overlap == 1)[0]
    final_detect_bbox = np.delete(detect_bbox, delet_overlap_index, axis=0)
    final_detect_mask = np.delete(detect_mask, delet_overlap_index, axis=2)
    final_detect_class_id = np.delete(detect_class_id, delet_overlap_index, axis=0)
    final_fusion_count = np.delete(fusion_count, delet_overlap_index, axis=0)
    final_detect_scores = np.delete(detect_scores, delet_overlap_index, axis=0) / final_fusion_count
    final_detect_slice_id = np.delete(detect_slice_id, delet_overlap_index, axis=0)



    return fusion_id, final_detect_bbox, final_detect_mask, \
           final_detect_class_id, final_detect_scores, final_detect_slice_id

def all_fusion_no_delet(rois, masks, class_ids, scores, slice_id, threshold=0.1, height_length=5, width_length=5):

    #             save_data["pre_bbox"] = detect_temp_bbox
    #             save_data["pre_mask"] = detect_temp_mask
    #             save_data["pre_class_id"] = detect_temp_class_id
    #             save_data["pre_score"] = detect_temp_scores
    #             save_data["pre_slice_id"] = detect_temp_slice_id

    height = rois[:, 2] - rois[:, 0]
    width = rois[:, 3] - rois[:, 1]
    center_y = rois[:, 0] + 0.5 * height
    center_x = rois[:, 1] + 0.5 * width

    instance_num = rois.shape[0]
    # 融合索引记录：-1为已融合需要保留的instance，other为融进other的instance，需要后续删除
    fusion_id = np.ones(shape=instance_num, dtype=int) * -1
    slice_index = []
    # 遍历所有的区域，找出区域的起始突触的Index
    for i in range(height_length * width_length):
        if len(np.where(slice_id == i)[0]) == 0:
            # 该区域没有突触实例
            if len(slice_index) != 0:
                # 分块列表存在数据
                slice_index.append(slice_index[-1])
            else:
                # 分块列表不存在数据，一直填充0
                slice_index.append(0)
        else:
            slice_index.append(np.where(slice_id == i)[0][0])
    # 加入中止标志
    slice_index.append(instance_num)

    assert instance_num == masks.shape[2] == class_ids.shape[0] == scores.shape[0] == \
           slice_id.shape[0], 'instance number must be equal ! '

    for i in range(height_length * width_length):
        # 遍历不同的子图，进行局部融合
        # 块内融合的索引序列
        compute_self_index = [index for index in range(slice_index[i], slice_index[i+1])]


        # 块间融合的索引序列
        if (i + 1) % width_length != 0 and i < height_length * width_length - width_length:
            # 当前块不为最后一列,且不为最后一行
            compute_other_index = [index for index in range(slice_index[i + 1], slice_index[i + 2])] + [index for index in range(
                slice_index[i + width_length], slice_index[i + width_length + 2])]
        elif (i + 1) % width_length != 0:
            # 当前块不为最后一列,但为最后一行
            compute_other_index = [index for index in range(slice_index[i + 1], slice_index[i + 2])]
        elif i < height_length * width_length - width_length:
            # 当前块为最后一列,但不为最后一行
            compute_other_index = [index for index in range(
                slice_index[i + width_length], slice_index[i + width_length + 1])]
        else:
            compute_other_index = []


        for j in compute_self_index:
            for k in compute_self_index:
                if j < k:
                    # 进行块间融合
                    if 2 * abs(center_y[j] - center_y[k]) < height[j] + height[k] and 2 * abs(center_x[j] - center_x[k]) < \
                            width[j] + width[k]:
                        # 找到最大的非背景区域
                        mask_y1 = min(rois[j, 0], rois[k, 0])
                        mask_y2 = max(rois[j, 2], rois[k, 2])
                        mask_x1 = min(rois[j, 1], rois[k, 1])
                        mask_x2 = max(rois[j, 3], rois[k, 3])
                        # 提取掩膜
                        masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j] > .5, (-1, 1)).astype(
                            np.float32)
                        masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, k] > .5, (-1, 1)).astype(
                            np.float32)
                        # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                        area1 = np.sum(masks1, axis=0)[0]
                        area2 = np.sum(masks2, axis=0)[0]

                        # intersections and union
                        # 计算点积,即交集,得到结果(instances1, instances2)
                        intersections = np.dot(masks1.T, masks2)[0][0]
                        # 计算并集
                        # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                        union = area1 + area2 - intersections
                        # 得到结果(instances1, instances2)的iou矩阵
                        overlaps = intersections / union
                        # 包围盒有重合，计算maskiou
                        if overlaps > threshold:
                            # k融入j中
                            index_fusion = j
                            index_delet = k
                            while fusion_id[index_fusion] != -1:
                                assert fusion_id[
                                           index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                                index_fusion = fusion_id[index_fusion]
                            # 需要融合的instance为需要融合的最小索引，可以直接融合
                            rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                            rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                            rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                            rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                            masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                            assert class_ids[index_fusion] == class_ids[
                                index_delet], 'you need to change this code to match the Multiclass classification'
                            scores[index_fusion] = max(scores[index_fusion], scores[index_delet])
                            fusion_id[index_delet] = index_fusion
            # 块间融合
            for k in compute_other_index:
                if 2 * abs(center_y[j] - center_y[k]) < height[j] + height[k] and 2 * abs(center_x[j] - center_x[k]) < \
                        width[j] + width[k]:
                    # 找到最大的非背景区域
                    mask_y1 = min(rois[j, 0], rois[k, 0])
                    mask_y2 = max(rois[j, 2], rois[k, 2])
                    mask_x1 = min(rois[j, 1], rois[k, 1])
                    mask_x2 = max(rois[j, 3], rois[k, 3])
                    # 提取掩膜
                    masks1 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, j] > .5, (-1, 1)).astype(
                        np.float32)
                    masks2 = np.reshape(masks[mask_y1:mask_y2, mask_x1:mask_x2, k] > .5, (-1, 1)).astype(
                        np.float32)
                    # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
                    area1 = np.sum(masks1, axis=0)[0]
                    area2 = np.sum(masks2, axis=0)[0]

                    # intersections and union
                    # 计算点积,即交集,得到结果(instances1, instances2)
                    intersections = np.dot(masks1.T, masks2)[0][0]
                    # 计算并集
                    # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
                    union = area1 + area2 - intersections
                    # 得到结果(instances1, instances2)的iou矩阵
                    overlaps = intersections / union
                    # 包围盒有重合，计算maskiou
                    if overlaps > threshold:
                        # k融入j中
                        index_fusion = j
                        index_delet = k
                        while fusion_id[index_fusion] != -1:
                            assert fusion_id[
                                       index_fusion] < index_fusion, "fusion_id[index_fusion] must smaller than index_fusion"
                            index_fusion = fusion_id[index_fusion]
                        # 需要融合的instance为需要融合的最小索引，可以直接融合
                        rois[index_fusion, 0] = min(rois[index_fusion, 0], rois[index_delet, 0])
                        rois[index_fusion, 1] = min(rois[index_fusion, 1], rois[index_delet, 1])
                        rois[index_fusion, 2] = max(rois[index_fusion, 2], rois[index_delet, 2])
                        rois[index_fusion, 3] = max(rois[index_fusion, 3], rois[index_delet, 3])
                        masks[:, :, index_fusion] = masks[:, :, index_fusion] + masks[:, :, index_delet]
                        assert class_ids[index_fusion] == class_ids[
                            index_delet], 'you need to change this code to match the Multiclass classification'
                        scores[index_fusion] = max(scores[index_fusion], scores[index_delet])
                        fusion_id[index_delet] = index_fusion

    # delet
    delet = np.where(fusion_id != -1)[0]
    detect_bbox = np.delete(rois, delet, axis=0)
    detect_mask = np.delete(masks, delet, axis=2)
    detect_class_id = np.delete(class_ids, delet, axis=0)
    detect_scores = np.delete(scores, delet, axis=0)
    detect_slice_id = np.delete(slice_id, delet, axis=0)
    return fusion_id, detect_bbox, detect_mask, detect_class_id, detect_scores, detect_slice_id


# 载入数据集信息
dataset_image_info = dataset_test.image_info

display_img_num = 80

# display_list = np.random.choice(display_img_num, 20)
# display_list = np.array([6,12,18,24,30,36,42,48,54,60,66,72])
display_list = np.linspace(0, 89, num=89, endpoint=False, dtype=np.int)

# 子图mAP,平均mAP-bbox,mask SLICE_BBOX_MAP SLICE_MASK_MAP
# 子图aug1,简单融合相加,mAP 平均mAP-bbox,mask SLICE_BBOX_FUSION_MAP SLICE_MASK_FUSION_MAP
# 子图aug2,概率相加阈值，mAP 平均mAP-bbox,mask SLICE_BBOX_THRESHOLD_MAP SLICE_MASK_THRESHOLD_MAP
# 大图overlap bbox重合，总和mAP-bbox,mask ALL_BBOX_BBOXOVERLAP_MAP ALL_MASK_BBOXOVERLAP_MAP
# 大图overlap NMS 总和mAP-bbox,mask ALL_BBOX_NMS_MAP ALL_MASK_NMS_MAP
SLICE_BBOX_MAP_temp = 0
SLICE_BBOX_MAP = []
SLICE_MASK_MAP_temp = 0
SLICE_MASK_MAP = []

SLICE_BBOX_FUSION_MAP_temp = 0
SLICE_BBOX_FUSION_MAP = []
SLICE_MASK_FUSION_MAP_temp = 0
SLICE_MASK_FUSION_MAP = []

SLICE_BBOX_THRESHOLD_MAP_temp = 0
SLICE_BBOX_THRESHOLD_MAP = []
SLICE_MASK_THRESHOLD_MAP_temp = 0
SLICE_MASK_THRESHOLD_MAP = []

SLICE_ALL_BBOX_DELET = []
SLICE_ALL_MASK_DELET = []
SLICE_ALL_BBOX_NO_DELET = []
SLICE_ALL_MASK_NO_DELET = []

SLICE_FUSION_ALL_BBOX_DELET = []
SLICE_FUSION_ALL_MASK_DELET = []
SLICE_FUSION_ALL_BBOX_NO_DELET = []
SLICE_FUSION_ALL_MASK_NO_DELET = []

SLICE_THRESHOLD_ALL_BBOX_DELET = []
SLICE_THRESHOLD_ALL_MASK_DELET = []
SLICE_THRESHOLD_ALL_BBOX_NO_DELET = []
SLICE_THRESHOLD_ALL_MASK_NO_DELET = []

save_path = r"fusion_outcome_v1"
os.makedirs(save_path, exist_ok=True)

out_img_num = display_list.shape[0]
start_img = 89
assert start_img >= 89
# 初始化存储信息
# temp_img = np.zeros(shape=(0, 2048, 3), dtype=np.uint8)
# temp_img_height = np.zeros(shape=(512, 0, 3), dtype=np.uint8)
# temp_class_id = np.zeros(shape=(0, ), dtype=np.uint32)
# temp_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
# temp_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)

aug_t_class_id = np.zeros(shape=(0, ), dtype=np.uint32)
aug_t_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
aug_t_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)
aug_t_scores = np.zeros(shape=(0, ), dtype=np.float32)
aug_t_slice_id = np.zeros(shape=(0, ), dtype=np.uint8)

aug_no_class_id = np.zeros(shape=(0, ), dtype=np.uint32)
aug_no_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
aug_no_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)
aug_no_scores = np.zeros(shape=(0, ), dtype=np.float32)
aug_no_slice_id = np.zeros(shape=(0, ), dtype=np.uint8)

aug_f_class_id = np.zeros(shape=(0, ), dtype=np.uint32)
aug_f_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
aug_f_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)
aug_f_scores = np.zeros(shape=(0, ), dtype=np.float32)
aug_f_slice_id = np.zeros(shape=(0, ), dtype=np.uint8)



assert display_img_num <= (len(dataset_image_info)/25), 'display_img_num too big'


for display_img in range(start_img - 89, out_img_num):
    # 开始随机的display_img张图片进行展示
    for display_id in range(5*5*display_list[display_img], 5*5*(display_list[display_img]+1)):
        item = dataset_image_info[display_id]
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test,
                                                                                           inference_config,
                                                                                           item['id'],
                                                                                           use_mini_mask=False)
        print('%d picture, %d slice, %d slice in all' % (display_list[display_img],
                                        item['position_width'] + item['position_height'] * (item['slice'] + 1),
                                        item['id']))
        # 测试时数据增强

        rows, cols = original_image.shape[:2]
        # r['rois'], r['masks'], r['class_ids'], r['scores']
        # 原图像结果
        results0 = model.detect([original_image], verbose=0)[0]
        # 顺时针旋转90度结果
        results1 = model.detect([np.flip(np.transpose(original_image, (1, 0, 2)), 1)], verbose=0)[0]
        # 顺时针旋转180度结果
        results2 = model.detect([np.flip(np.flip(original_image, 0), 1)], verbose=0)[0]
        # 顺时针旋转270度结果
        results3 = model.detect([np.flip(np.transpose(original_image, (1, 0, 2)), 0)], verbose=0)[0]

        # 旋转90度的box复原
        y1 = results1['rois'][:, 0].copy()
        x1 = results1['rois'][:, 1].copy()
        y2 = results1['rois'][:, 2].copy()
        x2 = results1['rois'][:, 3].copy()
        results1['rois'][:, 0] = cols - x2
        results1['rois'][:, 1] = y1
        results1['rois'][:, 2] = cols - x1
        results1['rois'][:, 3] = y2
        # 旋转90度的mask复原
        # 行列互换
        results1['masks'] = np.transpose(results1['masks'], (1, 0, 2))
        # 按axis=0翻转
        results1['masks'] = np.flip(results1['masks'], 0)

        # 旋转180度的box复原
        y1 = results2['rois'][:, 0].copy()
        x1 = results2['rois'][:, 1].copy()
        y2 = results2['rois'][:, 2].copy()
        x2 = results2['rois'][:, 3].copy()
        results2['rois'][:, 0] = rows - y2
        results2['rois'][:, 1] = cols - x2
        results2['rois'][:, 2] = rows - y1
        results2['rois'][:, 3] = cols - x1
        # 旋转180度的mask复原
        results2['masks'] = np.flip(results2['masks'], 1)
        results2['masks'] = np.flip(results2['masks'], 0)

        # 旋转270度的box复原
        y1 = results3['rois'][:, 0].copy()
        x1 = results3['rois'][:, 1].copy()
        y2 = results3['rois'][:, 2].copy()
        x2 = results3['rois'][:, 3].copy()
        results3['rois'][:, 0] = x1
        results3['rois'][:, 1] = rows - y2
        results3['rois'][:, 2] = x2
        results3['rois'][:, 3] = rows - y1
        # 旋转270度的mask复原
        results3['masks'] = np.transpose(results3['masks'], (1, 0, 2))
        results3['masks'] = np.flip(results3['masks'], 1)


        results0['rois'], results0['masks'], results0['class_ids'], results0['scores'] = \
            single_fusion(results0['rois'], results0['masks'], results0['class_ids'], results0['scores'], threshold=0.0)

        results1['rois'], results1['masks'], results1['class_ids'], results1['scores'] = \
            single_fusion(results1['rois'], results1['masks'], results1['class_ids'], results1['scores'], threshold=0.0)

        results2['rois'], results2['masks'], results2['class_ids'], results2['scores'] = \
            single_fusion(results2['rois'], results2['masks'], results2['class_ids'], results2['scores'], threshold=0.0)

        results3['rois'], results3['masks'], results3['class_ids'], results3['scores'] = \
            single_fusion(results3['rois'], results3['masks'], results3['class_ids'], results3['scores'], threshold=0.0)




        # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names,
        #                             figsize=(8, 8))
        #
        # visualize.display_instances(original_image, results0['rois'], results0['masks'], results0['class_ids'],
        #                             dataset_val.class_names,
        #                             results0['scores'], figsize=(8, 8))
        #
        # visualize.display_instances(original_image, results1['rois'], results1['masks'], results1['class_ids'],
        #                             dataset_val.class_names,
        #                             results1['scores'], figsize=(8, 8))
        #
        # visualize.display_instances(original_image, results2['rois'], results2['masks'], results2['class_ids'],
        #                             dataset_val.class_names,
        #                             results2['scores'], figsize=(8, 8))
        #
        # visualize.display_instances(original_image, results3['rois'], results3['masks'], results3['class_ids'],
        #                             dataset_val.class_names,
        #                             results3['scores'], figsize=(8, 8))

        # 合并测试时增强的输出结果
        r = {}
        ro = {}
        r['rois'] = np.concatenate((results0['rois'], results1['rois'], results2['rois'], results3['rois']), axis=0)
        r['masks'] = np.concatenate((results0['masks'], results1['masks'], results2['masks'], results3['masks']), axis=2)
        r['class_ids'] = np.concatenate((results0['class_ids'], results1['class_ids'], results2['class_ids'], results3['class_ids']), axis=0)
        r['scores'] = np.concatenate((results0['scores'], results1['scores'], results2['scores'], results3['scores']), axis=0)

        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
        #                             r['scores'], figsize=(8, 8))

        ro['rois'], ro['masks'], ro['class_ids'], ro['scores'] = aug_only_fusion(r['rois'],
                                                                        r['masks'],
                                                                        r['class_ids'],
                                                                        r['scores'],
                                                                        aug_num=4,
                                                                        aug_target=0.495,
                                                                        threshold=0.0)

        r['rois'], r['masks'], r['class_ids'], r['scores'] = aug_fusion(r['rois'],
                                                                        r['masks'],
                                                                        r['class_ids'],
                                                                        r['scores'],
                                                                        aug_num=4,
                                                                        aug_target=0.495,
                                                                        threshold=0.0)

        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
        #                             r['scores'], figsize=(8, 8))



        bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(gt_bbox, gt_class_id,
                                                             results0['rois'], results0['class_ids'],
                                                             results0['scores'])

        if gt_bbox.shape[0] == 0:
            SLICE_BBOX_MAP_temp = SLICE_BBOX_MAP_temp + 1
        else:
            SLICE_BBOX_MAP_temp = SLICE_BBOX_MAP_temp + bbox_AP


        print('before aug boxes fusion ap:', bbox_AP)

        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             results0['rois'], results0['class_ids'],
                                                             results0['scores'], results0['masks'])

        if gt_bbox.shape[0] == 0:
            SLICE_MASK_MAP_temp = SLICE_MASK_MAP_temp + 1
        else:
            SLICE_MASK_MAP_temp = SLICE_MASK_MAP_temp + AP

        print('before aug fusion ap:', AP)


        bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(gt_bbox, gt_class_id,
                                                                                      ro['rois'], ro['class_ids'],
                                                                                      ro['scores'])

        if gt_bbox.shape[0] == 0:
            SLICE_BBOX_FUSION_MAP_temp = SLICE_BBOX_FUSION_MAP_temp + 1
        else:
            SLICE_BBOX_FUSION_MAP_temp = SLICE_BBOX_FUSION_MAP_temp + bbox_AP

        print('after aug boxes only fusion ap:', bbox_AP)



        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             ro['rois'], ro['class_ids'],
                                                             ro['scores'], ro['masks'])

        if gt_bbox.shape[0] == 0:
            SLICE_MASK_FUSION_MAP_temp = SLICE_MASK_FUSION_MAP_temp + 1
        else:
            SLICE_MASK_FUSION_MAP_temp = SLICE_MASK_FUSION_MAP_temp + AP

        print('after aug only fusion ap:', AP)



        bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(gt_bbox, gt_class_id,
                                                                                      r['rois'], r['class_ids'],
                                                                                      r['scores'])

        if gt_bbox.shape[0] == 0:
            SLICE_BBOX_THRESHOLD_MAP_temp = SLICE_BBOX_THRESHOLD_MAP_temp + 1
        else:
            SLICE_BBOX_THRESHOLD_MAP_temp = SLICE_BBOX_THRESHOLD_MAP_temp + bbox_AP

        print('after aug boxes threshold fusion ap:', bbox_AP)



        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r['rois'], r['class_ids'],
                                                             r['scores'], r['masks'])

        if gt_bbox.shape[0] == 0:
            SLICE_MASK_THRESHOLD_MAP_temp = SLICE_MASK_THRESHOLD_MAP_temp + 1
        else:
            SLICE_MASK_THRESHOLD_MAP_temp = SLICE_MASK_THRESHOLD_MAP_temp + AP

        print('after aug threshold fusion ap:', AP)



        print('aug ok')

        # 子图的数据处理完毕，记录相关数据

        aug_t_slice_id = np.concatenate((aug_t_slice_id, np.ones(shape=len(r['rois']), dtype=np.uint8) * (
                item['position_height'] * (item['slice']+1) + item['position_width'])), axis=0)

        aug_no_slice_id = np.concatenate((aug_no_slice_id, np.ones(shape=len(results0['rois']), dtype=np.uint8) * (
                item['position_height'] * (item['slice']+1) + item['position_width'])), axis=0)

        aug_f_slice_id = np.concatenate((aug_f_slice_id, np.ones(shape=len(ro['rois']), dtype=np.uint8) * (
                item['position_height'] * (item['slice']+1) + item['position_width'])), axis=0)

        # 开始将该子图进行大图的融合
        if item['position_height'] == (item['slice']) and item['position_width'] == (item['slice']):
            # 最后一张子图
            ################################################
            all_img = cv2.imread(item['path_to_raw'], -1)
            all_img = cv2.resize(all_img, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
            all_img = np.stack((all_img, all_img, all_img), axis=2)

            label = cv2.imread(item['path_to_label'], -1)
            label = label.astype(np.uint8)
            label = cv2.resize(label, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
            label[label != 0] = 255

            # 寻找连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)
            # 取出当前所有连通域的面积
            out_area = stats[:, 4]
            # 根据连通域的大小进行对象的筛选
            out_sort = np.where((out_area > (2000 // 16)) & (out_area < (100000 // 16)))[0]
            # 取出剩余的连通域面积
            out_area = stats[out_sort, 4]
            # 根据连通域面积的从大到小取出对应的索引
            out_sort = out_sort[np.argsort(out_area)][::-1]
            count = len(out_sort)
            whole_mask = np.zeros([count, 2048, 2048], dtype=np.uint8)
            for mask_item in range(count):
                single_label = labels.copy().astype(np.uint8)
                single_label[single_label != out_sort[mask_item]] = 0
                single_label[single_label == out_sort[mask_item]] = 255
                whole_mask[mask_item] = single_label
            whole_mask = np.transpose(whole_mask, [1, 2, 0])
            whole_mask = whole_mask.astype(np.bool)
            # Map class names to class IDs.
            # 找到检测目标所属的id编号，全都是突触，所以构造全1数组
            whole_class_ids = np.ones((count,), dtype=np.int)

            whole_bbox = utils.extract_bboxes(whole_mask)

            ################################################

            # temp_class_id = np.concatenate((temp_class_id, gt_class_id), axis=0).astype(np.uint32)

            # gt_bbox[:, 0] = gt_bbox[:, 0] + item['position_height'] * 384
            # gt_bbox[:, 1] = gt_bbox[:, 1] + item['position_width'] * 384
            # gt_bbox[:, 2] = gt_bbox[:, 2] + item['position_height'] * 384
            # gt_bbox[:, 3] = gt_bbox[:, 3] + item['position_width'] * 384
            # temp_bbox = np.concatenate((temp_bbox, gt_bbox), axis=0).astype(np.uint32)


            # gt_mask = np.pad(gt_mask, ((item['position_height'] * 384, (item['slice'] * 512 - 512 - item['position_height'] * 384)),
            #                            (item['position_width'] * 384, (item['slice'] * 512 - 512 - item['position_width'] * 384)),
            #                            (0, 0)), 'constant', constant_values=False)
            # temp_mask = np.concatenate((temp_mask, gt_mask), axis=2).astype(np.bool)
            #############################################################
            # threshold 叠加
            aug_t_class_id = np.concatenate((aug_t_class_id, r['class_ids']), axis=0).astype(np.uint32)
            r['rois'][:, 0] = r['rois'][:, 0] + item['position_height'] * 384
            r['rois'][:, 1] = r['rois'][:, 1] + item['position_width'] * 384
            r['rois'][:, 2] = r['rois'][:, 2] + item['position_height'] * 384
            r['rois'][:, 3] = r['rois'][:, 3] + item['position_width'] * 384
            aug_t_bbox = np.concatenate((aug_t_bbox, r['rois']), axis=0).astype(np.uint32)
            aug_t_scores = np.concatenate((aug_t_scores, r['scores']), axis=0).astype(np.float32)
            r['masks'] = np.pad(r['masks'], ((item['position_height'] * 384, (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                       (item['position_width'] * 384, (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                       (0, 0)), 'constant', constant_values=False)
            aug_t_mask = np.concatenate((aug_t_mask, r['masks']), axis=2).astype(np.bool)

            # fusion叠加
            aug_f_class_id = np.concatenate((aug_f_class_id, ro['class_ids']), axis=0).astype(np.uint32)
            ro['rois'][:, 0] = ro['rois'][:, 0] + item['position_height'] * 384
            ro['rois'][:, 1] = ro['rois'][:, 1] + item['position_width'] * 384
            ro['rois'][:, 2] = ro['rois'][:, 2] + item['position_height'] * 384
            ro['rois'][:, 3] = ro['rois'][:, 3] + item['position_width'] * 384
            aug_f_bbox = np.concatenate((aug_f_bbox, ro['rois']), axis=0).astype(np.uint32)
            aug_f_scores = np.concatenate((aug_f_scores, ro['scores']), axis=0).astype(np.float32)
            ro['masks'] = np.pad(ro['masks'], ((item['position_height'] * 384, (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                       (item['position_width'] * 384, (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                       (0, 0)), 'constant', constant_values=False)
            aug_f_mask = np.concatenate((aug_f_mask, ro['masks']), axis=2).astype(np.bool)

            # no aug叠加
            aug_no_class_id = np.concatenate((aug_no_class_id, results0['class_ids']), axis=0).astype(np.uint32)
            results0['rois'][:, 0] = results0['rois'][:, 0] + item['position_height'] * 384
            results0['rois'][:, 1] = results0['rois'][:, 1] + item['position_width'] * 384
            results0['rois'][:, 2] = results0['rois'][:, 2] + item['position_height'] * 384
            results0['rois'][:, 3] = results0['rois'][:, 3] + item['position_width'] * 384
            aug_no_bbox = np.concatenate((aug_no_bbox, results0['rois']), axis=0).astype(np.uint32)
            aug_no_scores = np.concatenate((aug_no_scores, results0['scores']), axis=0).astype(np.float32)
            results0['masks'] = np.pad(results0['masks'], ((item['position_height'] * 384, (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                       (item['position_width'] * 384, (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                       (0, 0)), 'constant', constant_values=False)
            aug_no_mask = np.concatenate((aug_no_mask, results0['masks']), axis=2).astype(np.bool)


            SLICE_BBOX_MAP_temp = SLICE_BBOX_MAP_temp / 25
            SLICE_BBOX_MAP.append(SLICE_BBOX_MAP_temp)
            print('SLICE_BBOX_MAP:', SLICE_BBOX_MAP_temp)
            SLICE_BBOX_MAP_temp = 0

            SLICE_MASK_MAP_temp = SLICE_MASK_MAP_temp / 25
            SLICE_MASK_MAP.append(SLICE_MASK_MAP_temp)
            print('SLICE_MASK_MAP:', SLICE_MASK_MAP_temp)
            SLICE_MASK_MAP_temp = 0

            SLICE_BBOX_FUSION_MAP_temp = SLICE_BBOX_FUSION_MAP_temp / 25
            SLICE_BBOX_FUSION_MAP.append(SLICE_BBOX_FUSION_MAP_temp)
            print('SLICE_BBOX_FUSION_MAP:', SLICE_BBOX_FUSION_MAP_temp)
            SLICE_BBOX_FUSION_MAP_temp = 0

            SLICE_MASK_FUSION_MAP_temp = SLICE_MASK_FUSION_MAP_temp / 25
            SLICE_MASK_FUSION_MAP.append(SLICE_MASK_FUSION_MAP_temp)
            print('SLICE_MASK_FUSION_MAP:', SLICE_MASK_FUSION_MAP_temp)
            SLICE_MASK_FUSION_MAP_temp = 0

            SLICE_BBOX_THRESHOLD_MAP_temp = SLICE_BBOX_THRESHOLD_MAP_temp / 25
            SLICE_BBOX_THRESHOLD_MAP.append(SLICE_BBOX_THRESHOLD_MAP_temp)
            print('SLICE_BBOX_THRESHOLD_MAP:', SLICE_BBOX_THRESHOLD_MAP_temp)
            SLICE_BBOX_THRESHOLD_MAP_temp = 0

            SLICE_MASK_THRESHOLD_MAP_temp = SLICE_MASK_THRESHOLD_MAP_temp / 25
            SLICE_MASK_THRESHOLD_MAP.append(SLICE_MASK_THRESHOLD_MAP_temp)
            print('SLICE_MASK_THRESHOLD_MAP:', SLICE_MASK_THRESHOLD_MAP_temp)
            SLICE_MASK_THRESHOLD_MAP_temp = 0

            ########################################
            # threshold 大图删除操作
            _, aug_t_bbox_d, aug_t_mask_d, aug_t_class_id_d, aug_t_scores_d, aug_t_slice_id_d = \
                all_fusion(rois=aug_t_bbox,
                               masks=aug_t_mask,
                               class_ids=aug_t_class_id,
                               scores=aug_t_scores,
                               slice_id=aug_t_slice_id,
                               threshold=0.1,
                               height_length=5,
                               width_length=5)

            AP, precisions, recalls, overlaps = utils.compute_ap(whole_bbox, whole_class_ids, whole_mask,
                                                                 aug_t_bbox_d, aug_t_class_id_d,
                                                                 aug_t_scores_d, aug_t_mask_d)
            print('all aug threshold delet mask ap:', AP)
            SLICE_THRESHOLD_ALL_MASK_DELET.append(AP)
            bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(whole_bbox, whole_class_ids,
                                                                 aug_t_bbox_d, aug_t_class_id_d,
                                                                 aug_t_scores_d)
            print('all aug threshold delet bbox ap:', bbox_AP)
            SLICE_THRESHOLD_ALL_BBOX_DELET.append(bbox_AP)

            # threshold 大图未删除操作
            _, aug_t_bbox_nd, aug_t_mask_nd, aug_t_class_id_nd, aug_t_scores_nd, aug_t_slice_id_nd = \
                all_fusion_no_delet(rois=aug_t_bbox,
                               masks=aug_t_mask,
                               class_ids=aug_t_class_id,
                               scores=aug_t_scores,
                               slice_id=aug_t_slice_id,
                               threshold=0.1,
                               height_length=5,
                               width_length=5)

            AP, precisions, recalls, overlaps = utils.compute_ap(whole_bbox, whole_class_ids, whole_mask,
                                                                 aug_t_bbox_nd, aug_t_class_id_nd,
                                                                 aug_t_scores_nd, aug_t_mask_nd)
            print('all aug threshold no_delet mask ap:', AP)
            SLICE_THRESHOLD_ALL_MASK_NO_DELET.append(AP)

            bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(whole_bbox, whole_class_ids,
                                                                 aug_t_bbox_nd, aug_t_class_id_nd,
                                                                 aug_t_scores_nd)
            print('all aug threshold no_delet bbox ap:', bbox_AP)

            SLICE_THRESHOLD_ALL_BBOX_NO_DELET.append(bbox_AP)

            ########################################
            # fusion 大图删除操作
            _, aug_f_bbox_d, aug_f_mask_d, aug_f_class_id_d, aug_f_scores_d, aug_f_slice_id_d = \
                all_fusion(rois=aug_f_bbox,
                           masks=aug_f_mask,
                           class_ids=aug_f_class_id,
                           scores=aug_f_scores,
                           slice_id=aug_f_slice_id,
                           threshold=0.1,
                           height_length=5,
                           width_length=5)

            temp_save_bbox = aug_f_bbox_d
            temp_save_mask = aug_f_mask_d
            temp_save_class_id = aug_f_class_id_d
            temp_save_scores = aug_f_scores_d
            temp_save_slice_id = aug_f_slice_id_d


            AP, precisions, recalls, overlaps = utils.compute_ap(whole_bbox, whole_class_ids, whole_mask,
                                                                 aug_f_bbox_d, aug_f_class_id_d,
                                                                 aug_f_scores_d, aug_f_mask_d)
            print('all aug fusion delet mask ap:', AP)
            SLICE_FUSION_ALL_MASK_DELET.append(AP)
            bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(whole_bbox, whole_class_ids,
                                                                 aug_f_bbox_d, aug_f_class_id_d,
                                                                 aug_f_scores_d)
            print('all aug fusion delet bbox ap:', bbox_AP)
            SLICE_FUSION_ALL_BBOX_DELET.append(bbox_AP)



            # fusion 大图未删除操作
            _, aug_f_bbox_nd, aug_f_mask_nd, aug_f_class_id_nd, aug_f_scores_nd, aug_f_slice_id_nd = \
                all_fusion_no_delet(rois=aug_f_bbox,
                                    masks=aug_f_mask,
                                    class_ids=aug_f_class_id,
                                    scores=aug_f_scores,
                                    slice_id=aug_f_slice_id,
                                    threshold=0.1,
                                    height_length=5,
                                    width_length=5)

            AP, precisions, recalls, overlaps = utils.compute_ap(whole_bbox, whole_class_ids, whole_mask,
                                                                 aug_f_bbox_nd, aug_f_class_id_nd,
                                                                 aug_f_scores_nd, aug_f_mask_nd)
            print('all aug fusion no_delet mask ap:', AP)
            SLICE_FUSION_ALL_MASK_NO_DELET.append(AP)
            bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(whole_bbox, whole_class_ids,
                                                                 aug_f_bbox_nd, aug_f_class_id_nd,
                                                                 aug_f_scores_nd)
            print('all aug fusion no_delet bbox ap:', bbox_AP)
            SLICE_FUSION_ALL_BBOX_NO_DELET.append(bbox_AP)

            ########################################
            # no aug fusion 大图删除操作
            _, aug_no_bbox_d, aug_no_mask_d, aug_no_class_id_d, aug_no_scores_d, aug_no_slice_id_d = \
                all_fusion(rois=aug_no_bbox,
                           masks=aug_no_mask,
                           class_ids=aug_no_class_id,
                           scores=aug_no_scores,
                           slice_id=aug_no_slice_id,
                           threshold=0.1,
                           height_length=5,
                           width_length=5)

            AP, precisions, recalls, overlaps = utils.compute_ap(whole_bbox, whole_class_ids, whole_mask,
                                                                 aug_no_bbox_d, aug_no_class_id_d,
                                                                 aug_no_scores_d, aug_no_mask_d)
            print('all no_aug delet mask ap:', AP)
            SLICE_ALL_MASK_DELET.append(AP)
            bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(whole_bbox, whole_class_ids,
                                                                 aug_no_bbox_d, aug_no_class_id_d,
                                                                 aug_no_scores_d)
            print('all no_aug delet bbox ap:', bbox_AP)
            SLICE_ALL_BBOX_DELET.append(bbox_AP)



            # no aug fusion 大图未删除操作
            _, aug_no_bbox_nd, aug_no_mask_nd, aug_no_class_id_nd, aug_no_scores_nd, aug_no_slice_id_nd = \
                all_fusion_no_delet(rois=aug_no_bbox,
                                    masks=aug_no_mask,
                                    class_ids=aug_no_class_id,
                                    scores=aug_no_scores,
                                    slice_id=aug_no_slice_id,
                                    threshold=0.1,
                                    height_length=5,
                                    width_length=5)
            AP, precisions, recalls, overlaps = utils.compute_ap(whole_bbox, whole_class_ids, whole_mask,
                                                                 aug_no_bbox_nd, aug_no_class_id_nd,
                                                                 aug_no_scores_nd, aug_no_mask_nd)
            print('all no_aug no_delet mask ap:', AP)
            SLICE_ALL_MASK_NO_DELET.append(AP)
            bbox_AP, bbox_precisions, bbox_recalls, bbox_overlaps = utils.compute_bbox_ap(whole_bbox, whole_class_ids,
                                                                 aug_no_bbox_nd, aug_no_class_id_nd,
                                                                 aug_no_scores_nd)
            print('all no_aug no_delet bbox ap:', bbox_AP)
            SLICE_ALL_BBOX_NO_DELET.append(bbox_AP)


            print("now SLICE_BBOX_MAP: ", np.mean(SLICE_BBOX_MAP))
            print("now SLICE_MASK_MAP: ", np.mean(SLICE_MASK_MAP))
            print("now SLICE_BBOX_FUSION_MAP: ", np.mean(SLICE_BBOX_FUSION_MAP))
            print("now SLICE_MASK_FUSION_MAP: ", np.mean(SLICE_MASK_FUSION_MAP))
            print("now SLICE_BBOX_THRESHOLD_MAP: ", np.mean(SLICE_BBOX_THRESHOLD_MAP))
            print("now SLICE_MASK_THRESHOLD_MAP: ", np.mean(SLICE_MASK_THRESHOLD_MAP))
            print("now SLICE_ALL_BBOX_DELET: ", np.mean(SLICE_ALL_BBOX_DELET))
            print("now SLICE_ALL_MASK_DELET: ", np.mean(SLICE_ALL_MASK_DELET))
            print("now SLICE_ALL_BBOX_NO_DELET: ", np.mean(SLICE_ALL_BBOX_NO_DELET))
            print("now SLICE_ALL_MASK_NO_DELET: ", np.mean(SLICE_ALL_MASK_NO_DELET))
            print("now SLICE_FUSION_ALL_BBOX_DELET: ", np.mean(SLICE_FUSION_ALL_BBOX_DELET))
            print("now SLICE_FUSION_ALL_MASK_DELET: ", np.mean(SLICE_FUSION_ALL_MASK_DELET))
            print("now SLICE_FUSION_ALL_BBOX_NO_DELET: ", np.mean(SLICE_FUSION_ALL_BBOX_NO_DELET))
            print("now SLICE_FUSION_ALL_MASK_NO_DELET: ", np.mean(SLICE_FUSION_ALL_MASK_NO_DELET))
            print("now SLICE_THRESHOLD_ALL_BBOX_DELET: ", np.mean(SLICE_THRESHOLD_ALL_BBOX_DELET))
            print("now SLICE_THRESHOLD_ALL_MASK_DELET: ", np.mean(SLICE_THRESHOLD_ALL_MASK_DELET))
            print("now SLICE_THRESHOLD_ALL_BBOX_NO_DELET: ", np.mean(SLICE_THRESHOLD_ALL_BBOX_NO_DELET))
            print("now SLICE_THRESHOLD_ALL_MASK_NO_DELET: ", np.mean(SLICE_THRESHOLD_ALL_MASK_NO_DELET))

            print('all ok')
            save_data = {}

            save_data["img"] = all_img
            save_data["gt_bbox"] = whole_bbox
            save_data["gt_mask"] = whole_mask
            save_data["gt_class_id"] = whole_class_ids
            # save_data["gt_slice_id"] = temp_slice_id
            save_data["class_names"] = dataset_train.class_names

            save_data["pre_bbox"] = temp_save_bbox
            save_data["pre_mask"] = temp_save_mask
            save_data["pre_class_id"] = temp_save_class_id
            save_data["pre_score"] = temp_save_scores
            save_data["pre_slice_id"] = temp_save_slice_id

            print('save data  ok')

            # save
            cur_outcome_path = os.path.join(save_path, "%04d.pickle" % start_img)
            with open(cur_outcome_path, 'wb') as f_out:
                joblib.dump(save_data, f_out)
                print(
                    'save_data has been written to {}, and can be loaded when testing to ensure correct results'.format(
                        cur_outcome_path))


            # with open(cur_outcome_path, 'rb') as f_in:
            #     classes_count = pickle.load(f_in)
            #     print('classes_count has been read from {}'.format(cur_outcome_path))


            # visualize.display_instances(temp_img, np.zeros(shape=(0, 4), dtype=np.uint32),
            #                             np.zeros(shape=(2048, 2048, 0), dtype=np.bool),
            #                             np.zeros(shape=(0,), dtype=np.uint32),
            #                             dataset_train.class_names,
            #                             figsize=(8, 8))
            #
            #
            # visualize.display_instances(temp_img, temp_bbox, temp_mask, temp_class_id, dataset_train.class_names,
            #                             figsize=(8, 8))
            #
            # visualize.display_instances(temp_img, detect_temp_bbox, detect_temp_mask, detect_temp_class_id, dataset_val.class_names,
            #                             detect_temp_scores, figsize=(8, 8))


            start_img = start_img + 1

            aug_t_class_id = np.zeros(shape=(0,), dtype=np.uint32)
            aug_t_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
            aug_t_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)
            aug_t_scores = np.zeros(shape=(0,), dtype=np.float32)
            aug_t_slice_id = np.zeros(shape=(0,), dtype=np.uint8)

            aug_no_class_id = np.zeros(shape=(0,), dtype=np.uint32)
            aug_no_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
            aug_no_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)
            aug_no_scores = np.zeros(shape=(0,), dtype=np.float32)
            aug_no_slice_id = np.zeros(shape=(0,), dtype=np.uint8)

            aug_f_class_id = np.zeros(shape=(0,), dtype=np.uint32)
            aug_f_bbox = np.zeros(shape=(0, 4), dtype=np.uint32)
            aug_f_mask = np.zeros(shape=(2048, 2048, 0), dtype=np.bool)
            aug_f_scores = np.zeros(shape=(0,), dtype=np.float32)
            aug_f_slice_id = np.zeros(shape=(0,), dtype=np.uint8)

        else:
            if item['position_width'] == (item['slice']):

                ######################################################################
                # threshold 叠加
                aug_t_class_id = np.concatenate((aug_t_class_id, r['class_ids']), axis=0).astype(np.uint32)
                r['rois'][:, 0] = r['rois'][:, 0] + item['position_height'] * 384
                r['rois'][:, 1] = r['rois'][:, 1] + item['position_width'] * 384
                r['rois'][:, 2] = r['rois'][:, 2] + item['position_height'] * 384
                r['rois'][:, 3] = r['rois'][:, 3] + item['position_width'] * 384
                aug_t_bbox = np.concatenate((aug_t_bbox, r['rois']), axis=0).astype(np.uint32)
                aug_t_scores = np.concatenate((aug_t_scores, r['scores']), axis=0).astype(np.float32)
                r['masks'] = np.pad(r['masks'],
                                    ((item['position_height'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                  (item['position_width'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                    (0, 0)), 'constant', constant_values=False)
                aug_t_mask = np.concatenate((aug_t_mask, r['masks']), axis=2).astype(np.bool)

                # fusion 叠加
                aug_f_class_id = np.concatenate((aug_f_class_id, ro['class_ids']), axis=0).astype(np.uint32)
                ro['rois'][:, 0] = ro['rois'][:, 0] + item['position_height'] * 384
                ro['rois'][:, 1] = ro['rois'][:, 1] + item['position_width'] * 384
                ro['rois'][:, 2] = ro['rois'][:, 2] + item['position_height'] * 384
                ro['rois'][:, 3] = ro['rois'][:, 3] + item['position_width'] * 384
                aug_f_bbox = np.concatenate((aug_f_bbox, ro['rois']), axis=0).astype(np.uint32)
                aug_f_scores = np.concatenate((aug_f_scores, ro['scores']), axis=0).astype(np.float32)
                ro['masks'] = np.pad(ro['masks'],
                                    ((item['position_height'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                  (item['position_width'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                    (0, 0)), 'constant', constant_values=False)
                aug_f_mask = np.concatenate((aug_f_mask, ro['masks']), axis=2).astype(np.bool)

                # no aug 叠加
                aug_no_class_id = np.concatenate((aug_no_class_id, results0['class_ids']), axis=0).astype(np.uint32)
                results0['rois'][:, 0] = results0['rois'][:, 0] + item['position_height'] * 384
                results0['rois'][:, 1] = results0['rois'][:, 1] + item['position_width'] * 384
                results0['rois'][:, 2] = results0['rois'][:, 2] + item['position_height'] * 384
                results0['rois'][:, 3] = results0['rois'][:, 3] + item['position_width'] * 384
                aug_no_bbox = np.concatenate((aug_no_bbox, results0['rois']), axis=0).astype(np.uint32)
                aug_no_scores = np.concatenate((aug_no_scores, results0['scores']), axis=0).astype(np.float32)
                results0['masks'] = np.pad(results0['masks'],
                                    ((item['position_height'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                  (item['position_width'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                    (0, 0)), 'constant', constant_values=False)
                aug_no_mask = np.concatenate((aug_no_mask, results0['masks']), axis=2).astype(np.bool)

            else:

                ####################################################################
                # threshold 叠加
                aug_t_class_id = np.concatenate((aug_t_class_id, r['class_ids']), axis=0).astype(np.uint32)
                r['rois'][:, 0] = r['rois'][:, 0] + item['position_height'] * 384
                r['rois'][:, 1] = r['rois'][:, 1] + item['position_width'] * 384
                r['rois'][:, 2] = r['rois'][:, 2] + item['position_height'] * 384
                r['rois'][:, 3] = r['rois'][:, 3] + item['position_width'] * 384
                aug_t_bbox = np.concatenate((aug_t_bbox, r['rois']), axis=0).astype(np.uint32)
                aug_t_scores = np.concatenate((aug_t_scores, r['scores']), axis=0).astype(np.float32)
                r['masks'] = np.pad(r['masks'], ((item['position_height'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                  (item['position_width'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                (0, 0)), 'constant', constant_values=False)
                aug_t_mask = np.concatenate((aug_t_mask, r['masks']), axis=2).astype(np.bool)

                # fusion 叠加
                aug_f_class_id = np.concatenate((aug_f_class_id, ro['class_ids']), axis=0).astype(np.uint32)
                ro['rois'][:, 0] = ro['rois'][:, 0] + item['position_height'] * 384
                ro['rois'][:, 1] = ro['rois'][:, 1] + item['position_width'] * 384
                ro['rois'][:, 2] = ro['rois'][:, 2] + item['position_height'] * 384
                ro['rois'][:, 3] = ro['rois'][:, 3] + item['position_width'] * 384
                aug_f_bbox = np.concatenate((aug_f_bbox, ro['rois']), axis=0).astype(np.uint32)
                aug_f_scores = np.concatenate((aug_f_scores, ro['scores']), axis=0).astype(np.float32)
                ro['masks'] = np.pad(ro['masks'], ((item['position_height'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                  (item['position_width'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                (0, 0)), 'constant', constant_values=False)
                aug_f_mask = np.concatenate((aug_f_mask, ro['masks']), axis=2).astype(np.bool)

                # threshold 叠加
                aug_no_class_id = np.concatenate((aug_no_class_id, results0['class_ids']), axis=0).astype(np.uint32)
                results0['rois'][:, 0] = results0['rois'][:, 0] + item['position_height'] * 384
                results0['rois'][:, 1] = results0['rois'][:, 1] + item['position_width'] * 384
                results0['rois'][:, 2] = results0['rois'][:, 2] + item['position_height'] * 384
                results0['rois'][:, 3] = results0['rois'][:, 3] + item['position_width'] * 384
                aug_no_bbox = np.concatenate((aug_no_bbox, results0['rois']), axis=0).astype(np.uint32)
                aug_no_scores = np.concatenate((aug_no_scores, results0['scores']), axis=0).astype(np.float32)
                results0['masks'] = np.pad(results0['masks'], ((item['position_height'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_height'] * 384)),
                                  (item['position_width'] * 384,
                                   (item['slice'] * 512 - 512 - item['position_width'] * 384)),
                                (0, 0)), 'constant', constant_values=False)
                aug_no_mask = np.concatenate((aug_no_mask, results0['masks']), axis=2).astype(np.bool)


VAL_DICT = {}
VAL_DICT['SLICE_BBOX_MAP'] = SLICE_BBOX_MAP
VAL_DICT['SLICE_MASK_MAP'] = SLICE_MASK_MAP
VAL_DICT['SLICE_BBOX_FUSION_MAP'] = SLICE_BBOX_FUSION_MAP
VAL_DICT['SLICE_MASK_FUSION_MAP'] = SLICE_MASK_FUSION_MAP
VAL_DICT['SLICE_BBOX_THRESHOLD_MAP'] = SLICE_BBOX_THRESHOLD_MAP
VAL_DICT['SLICE_MASK_THRESHOLD_MAP'] = SLICE_MASK_THRESHOLD_MAP
VAL_DICT['SLICE_ALL_BBOX_DELET'] = SLICE_ALL_BBOX_DELET
VAL_DICT['SLICE_ALL_MASK_DELET'] = SLICE_ALL_MASK_DELET
VAL_DICT['SLICE_ALL_BBOX_NO_DELET'] = SLICE_ALL_BBOX_NO_DELET
VAL_DICT['SLICE_ALL_MASK_NO_DELET'] = SLICE_ALL_MASK_NO_DELET
VAL_DICT['SLICE_FUSION_ALL_BBOX_DELET'] = SLICE_FUSION_ALL_BBOX_DELET
VAL_DICT['SLICE_FUSION_ALL_MASK_DELET'] = SLICE_FUSION_ALL_MASK_DELET
VAL_DICT['SLICE_FUSION_ALL_BBOX_NO_DELET'] = SLICE_FUSION_ALL_BBOX_NO_DELET
VAL_DICT['SLICE_FUSION_ALL_MASK_NO_DELET'] = SLICE_FUSION_ALL_MASK_NO_DELET
VAL_DICT['SLICE_THRESHOLD_ALL_BBOX_DELET'] = SLICE_THRESHOLD_ALL_BBOX_DELET
VAL_DICT['SLICE_THRESHOLD_ALL_MASK_DELET'] = SLICE_THRESHOLD_ALL_MASK_DELET
VAL_DICT['SLICE_THRESHOLD_ALL_BBOX_NO_DELET'] = SLICE_THRESHOLD_ALL_BBOX_NO_DELET
VAL_DICT['SLICE_THRESHOLD_ALL_MASK_NO_DELET'] = SLICE_THRESHOLD_ALL_MASK_NO_DELET


with open("mAP_info.pickle", 'wb') as f_out:
    joblib.dump(VAL_DICT, f_out)
    print(
        'save_data has been written to {}, and can be loaded when testing to ensure correct results'.format(
            "mAP_info.pickle"))


print('OK')