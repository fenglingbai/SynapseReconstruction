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
import matplotlib.pyplot as plt

# # 采用CPU训练
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'                  # 指定第一块GPU可用
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True                    # 程序按需申请内存
# sess = tf.Session(config=config)

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

# 保存logs与trained model的文件夹
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# pretrained model path
COCO_MODEL_PATH = r"D:\guojy\GJY_mrcnn004_aug_aug_usecoco\mask_rcnn_coco.h5"
# data path
DATA_DIR = "data/path"
RAW_DIR = os.path.join(DATA_DIR, "raw")
LABEL_DIR = os.path.join(DATA_DIR, "label16")


# os.path.exists(COCO_MODEL_PATH)
# ## Configurations
########################################################################################################################



class SynapsesConfig(Config):
    # 参数配置辨识名
    NAME = "synapses"
    # 在一块gpu上训练，每块GPU训练一张图片，Batch size 为1 (GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # 类别数量 (包含背景)
    NUM_CLASSES = 1 + 1  # 背景 + 突触

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


# ## Dataset
#
# Create a synthetic dataset
#
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
#
# * load_image()
# * load_mask()
# * image_reference()



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
            for j in range(slice):
                for k in range(slice):
                    # 将该图片信息录入到image_info中
                    # 完成后的形式为self.image_info =
                    # [{"id": 0,"source":"shapes","path":None,...},
                    #  {"id": 1,"source":"shapes","path":None,...},
                    #  ...
                    #  {"id": i,"source":"shapes","path":None,...}]
                    self.add_image("EMdata", image_id=i*16+j*4+k, path=None,
                                   path_to_raw=os.path.join(raw_path, raw_path_list[i+start]),
                                   path_to_label=os.path.join(label_path, label_path_list[i+start]),
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
        img = cv2.resize(img, dsize=(height*slice, width*slice), interpolation=cv2.INTER_CUBIC)
        out_img = img[j * height:(j + 1) * height, k * width:(k + 1) * width]
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
        temp_label = label[j * height:(j + 1) * height, k * width:(k + 1) * width]

        # 寻找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp_label, connectivity=8)
        # 取出当前所有连通域的面积
        out_area = stats[:, 4]
        # 根据连通域的大小进行对象的筛选
        out_sort = np.where((out_area > 100) & (out_area < 6000))[0]
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
        if count>0:
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

# ## Create Model

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


callbacks = modellib.get_callbacks(logging_file=config.LOGGING_FILE,
                          learning_rate_drop=config.LEARNING_RATE_DROP,
                          learning_rate_patience=config.LEARNING_RATE_PATIENCE,
                          early_stopping_patience=config.EARLY_STOPPING_PATIENCE)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            custom_callbacks=callbacks,
            augment=True,
            epochs=65,
            layers='heads')

print("OK")

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 2,
            custom_callbacks=callbacks,
            augment=True,
            epochs=300,
            layers="all")

print("All OK")

class InferenceConfig(SynapsesConfig):
    # 推理参数设置，这里设置为一张一张推理
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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


def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("保存txt文件成功")



img_list = np.random.choice(dataset_val.image_ids, 80)
APs = []
count1 = 0

# 遍历测试集
for image_id in img_list:
    # 加载测试集的ground truth
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    # 将所有ground truth载入并保存
    if count1 == 0:
        save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
    else:
        save_box = np.concatenate((save_box, gt_bbox), axis=0)
        save_class = np.concatenate((save_class, gt_class_id), axis=0)
        save_mask = np.concatenate((save_mask, gt_mask), axis=2)

    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

    # 启动检测
    results = model.detect([image], verbose=0)
    r = results[0]

    # 将所有检测结果保存
    if count1 == 0:
        save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
    else:
        save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
        save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
        save_score = np.concatenate((save_score, r["scores"]), axis=0)
        save_m = np.concatenate((save_m, r['masks']), axis=2)

    count1 += 1
    print("dataset_val:", count1)

# 计算AP, precision, recall
AP, precisions, recalls, overlaps = \
    utils.compute_ap(save_box, save_class, save_mask,
                     save_roi, save_id, save_score, save_m)

print("dataset_val AP: ", AP)
print("dataset_val mAP: ", np.mean(AP))

# 绘制PR曲线
plt.plot(recalls, precisions, 'b', label='PR')
plt.title('dataset_val precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# 保存precision, recall信息用于后续绘制图像
text_save('dataset_val_Kpreci.txt', precisions)
text_save('dataset_val_Krecall.txt', recalls)

print("dataset_val OK")



img_list = np.random.choice(dataset_train.image_ids, 80)
APs = []
count1 = 0

# 遍历测试集
for image_id in img_list:
    # 加载测试集的ground truth
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_train, inference_config,
                               image_id, use_mini_mask=False)
    # 将所有ground truth载入并保存
    if count1 == 0:
        save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
    else:
        save_box = np.concatenate((save_box, gt_bbox), axis=0)
        save_class = np.concatenate((save_class, gt_class_id), axis=0)
        save_mask = np.concatenate((save_mask, gt_mask), axis=2)

    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

    # 启动检测
    results = model.detect([image], verbose=0)
    r = results[0]

    # 将所有检测结果保存
    if count1 == 0:
        save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
    else:
        save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
        save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
        save_score = np.concatenate((save_score, r["scores"]), axis=0)
        save_m = np.concatenate((save_m, r['masks']), axis=2)

    count1 += 1
    print("dataset_train:", count1)

# 计算AP, precision, recall
AP, precisions, recalls, overlaps = \
    utils.compute_ap(save_box, save_class, save_mask,
                     save_roi, save_id, save_score, save_m)

print("dataset_train AP: ", AP)
print("dataset_train mAP: ", np.mean(AP))

# 绘制PR曲线
plt.plot(recalls, precisions, 'b', label='PR')
plt.title('dataset_train precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# 保存precision, recall信息用于后续绘制图像
text_save('dataset_train_Kpreci.txt', precisions)
text_save('dataset_train_Krecall.txt', recalls)

print("dataset_train OK")


