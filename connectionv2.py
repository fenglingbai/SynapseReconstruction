import cv2
import copy
import numpy as np
from skimage import io
#---------------------------------------basic utils------------------------------------------------

def extract_bboxes(mask):
    """
    功能: 根据mask信息提取bbox数据
    mask: [height, width, num_instances]. np.bool
    返回: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
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
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def compute_centroid(mask):
    """
    功能: 计算mask质心
    mask: [height, width]. np.array
    返回: y_index, x_index
    """
    weight_half = np.sum(mask) // 2
    # y centroid
    area_count = 0
    y_index = 0
    while(area_count < weight_half):
        area_count += np.sum(mask[y_index, :])
        y_index += 1
    # x centroid
    area_count = 0
    x_index = 0
    while(area_count < weight_half):
        area_count += np.sum(mask[:, x_index])
        x_index += 1

    return y_index, x_index

#---------------------------------------basic functions------------------------------------------------
def find_object(image, area_min=0, area_max=0, delet_biggest=True):
    """
    功能：提取单张图像的连通域实例对象
    image: 需要找连通域的二值图像
    area_min: 连通域的最小阈值, 默认0
    area_max: 连通域的最大阈值, 默认图像大小
    delet_biggest: 是否删除最大连通域(背景), 默认删除
    返回: 字典single_layer
    single_layer['mask'] mask array [height, width, num_instances] np.bool
    single_layer['bbox'] bbox array [num_instances, (y1, x1, y2, x2)] np.int32
    """
    ##########################################
    # 1.处理数据判断异常
    # 二值图，两个维度
    assert len(image.shape) == 2
    # 防止后续数据变动改变原始数据
    image_label = image.copy()
    image_label = image_label.astype(np.uint8)
    image_label[image_label != 0] = 255
    if area_max == 0:
        # 最大阈值为缺省值，默认设置为图像大小
        area_max = image.shape[0] * image.shape[1]
    ###########################################
    # 2.初始化返回参数
    single_layer = {}
    ###########################################
    # 3.提取连通域对象
    # 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_label, connectivity=8)
    # 取出当前所有连通域的面积
    out_area = stats[:, 4]
    # 根据连通域的大小进行对象的筛选
    out_sort = np.where((out_area > area_min) & (out_area < area_max))[0]
    # 取出剩余的连通域面积
    out_area = stats[out_sort, 4]
    # 根据连通域面积的从大到小取出对应的索引
    out_sort = out_sort[np.argsort(out_area)][::-1]
    if delet_biggest:
        # 删除背景
        out_sort = out_sort[1:]
    # 计算连通域数量
    count = len(out_sort)
    mask = np.zeros([count, image.shape[0], image.shape[1]], dtype=np.bool)
    for item in range(count):
        single_label = np.zeros([image.shape[0], image.shape[1]], dtype=np.bool)
        single_label[labels == out_sort[item]] = True
        mask[item] = single_label
    mask = np.transpose(mask, [1, 2, 0])
    if count>0:
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.bool)
        for i in range(count - 2, -1, -1):
            # 从count - 2倒序至0
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

    single_layer['mask'] = mask.astype(np.bool)
    single_layer['bbox'] = extract_bboxes(mask)

    # # 可视化代码，测试用，此处注释
    # draw_img = mask[:, :, 0].copy().astype(np.uint8)
    # draw_img = draw_img * 255
    # cv2.namedWindow("Airy disks", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("Airy disks", draw_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return single_layer

#---------------------------------------Similarity Metrics------------------------------------------------
def bbox_metrics(low_size, high_size, low_bbox, high_bbox, thre):
    """
    功能: bbox IoU
    low_bbox & high_bbox: 掩膜1和掩膜2包围盒坐标(y1, x1, y2, x2)
    low_mask & high_mask: 掩膜1和掩膜2在原图尺寸的掩膜信息
    thre: 连接设定阈值
    返回: 连接判定 True or False
    """
    # 计算bbox面积
    low_area = low_size[0] * low_size[1]
    high_area = high_size[0] * high_size[1]

    y1_max = max(low_bbox[0], high_bbox[0])
    y2_min = min(low_bbox[2], high_bbox[2])
    x1_max = max(low_bbox[1], high_bbox[1])
    x2_min = min(low_bbox[3], high_bbox[3])
    intersection = max(x2_min - x1_max, 0) * max(y2_min - y1_max, 0)
    union = low_area + high_area - intersection
    overlaps = intersection / union
    if overlaps > thre:
        return True
    else:
        return False

def mask0_metrics(low_bbox, high_bbox, low_mask, high_mask, thre):
    """
    功能: 掩膜IoU
    low_bbox & high_bbox: 掩膜1和掩膜2包围盒坐标(y1, x1, y2, x2)
    low_mask & high_mask: 掩膜1和掩膜2在原图尺寸的掩膜信息
    thre: 连接设定阈值
    返回: 连接判定 True or False
    """
    y1_min = min(low_bbox[0], high_bbox[0])
    x1_min = min(low_bbox[1], high_bbox[1])
    y2_max = max(low_bbox[2], high_bbox[2])
    x2_max = max(low_bbox[3], high_bbox[3])
    # 提取局部掩膜
    mask1 = low_mask[y1_min:y2_max, x1_min:x2_max]
    mask2 = high_mask[y1_min:y2_max, x1_min:x2_max]
    
    # 计算iou
    mask1 = np.reshape(mask1, (-1, 1)).astype(np.float32)
    mask2 = np.reshape(mask2, (-1, 1)).astype(np.float32)
    # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
    area1 = np.sum(mask1, axis=0)[0]
    area2 = np.sum(mask2, axis=0)[0]

    # intersections and union
    # 计算点积,即交集,得到结果(instances1, instances2)
    intersections = np.dot(mask1.T, mask2)[0][0]
    # 计算并集

    # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
    union = area1 + area2 - intersections
    # 得到结果(instances1, instances2)的iou矩阵
    overlaps = intersections / union
    if overlaps > thre:
        return True
    else:
        return False


def mask1_metrics(y_distance, x_distance, low_size, high_size, low_bbox, high_bbox, low_mask, high_mask, thre, reduce_ratio):
    """
    功能: 基于包围盒中心距离的掩膜IoU
    y_distance & x_distance: 包围盒中心y距离和x距离
    low_size & high_size: 掩膜1和掩膜2包围盒的尺寸
    low_bbox & high_bbox: 掩膜1和掩膜2包围盒坐标(y1, x1, y2, x2)
    low_mask & high_mask: 掩膜1和掩膜2在原图尺寸的掩膜信息
    thre: 连接设定阈值
    reduce_ratio: 相似度度量的衰减系数
    返回: 连接判定 True or False
    """
    distence = (y_distance ** 2 + x_distance ** 2) / (
                (low_size[0] + high_size[0]) ** 2 + (low_size[1] + high_size[1]) ** 2)
    # 提取局部掩膜
    mask1 = low_mask[low_bbox[0]:low_bbox[2], low_bbox[1]:low_bbox[3]]
    mask2 = high_mask[high_bbox[0]:high_bbox[2], high_bbox[1]:high_bbox[3]]
    # 向周围扩展至相同大小
    # 行扩充
    if low_size[0] > high_size[0]:
        # j_item_index，即mask2需要扩展
        height_pad = low_size[0] - high_size[0]
        pad_before = height_pad // 2
        pad_after = height_pad - pad_before
        # 默认填充False
        mask2 = np.pad(mask2, ((pad_before, pad_after), (0, 0)), 'constant')
    elif low_size[0] < high_size[0]:
        # i_item_index，即masks1需要扩展
        height_pad = high_size[0] - low_size[0]
        pad_before = height_pad // 2
        pad_after = height_pad - pad_before
        mask1 = np.pad(mask1, ((pad_before, pad_after), (0, 0)), 'constant')
    # 列扩充
    if low_size[1] > high_size[1]:
        # j_item_index，即masks2需要扩展
        width_pad = low_size[1] - high_size[1]
        pad_before = width_pad // 2
        pad_after = width_pad - pad_before
        mask2 = np.pad(mask2, ((0, 0), (pad_before, pad_after)), 'constant')
    elif low_size[1] < high_size[1]:
        # i_item_index，即masks1需要扩展
        width_pad = high_size[1] - low_size[1]
        pad_before = width_pad // 2
        pad_after = width_pad - pad_before
        mask1 = np.pad(mask1, ((0, 0), (pad_before, pad_after)), 'constant')

    # 计算iou
    mask1 = np.reshape(mask1, (-1, 1)).astype(np.float32)
    mask2 = np.reshape(mask2, (-1, 1)).astype(np.float32)
    # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
    area1 = np.sum(mask1, axis=0)[0]
    area2 = np.sum(mask2, axis=0)[0]

    # intersections and union
    # 计算点积,即交集,得到结果(instances1, instances2)
    intersections = np.dot(mask1.T, mask2)[0][0]
    # 计算并集

    # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
    union = area1 + area2 - intersections
    # 得到结果(instances1, instances2)的iou矩阵
    overlaps = intersections * (1 - min(reduce_ratio * distence, 1)) / union
    if overlaps > thre:
        return True
    else:
        return False


def mask2_metrics(low_size, high_size, low_bbox, high_bbox, low_mask, high_mask, thre, reduce_ratio):
    """
    功能: 基于掩膜质心距离的掩膜IoU
    low_size & high_size: 掩膜1和掩膜2包围盒的尺寸
    low_bbox & high_bbox: 掩膜1和掩膜2包围盒坐标(y1, x1, y2, x2)
    low_mask & high_mask: 掩膜1和掩膜2在原图尺寸的掩膜信息
    thre: 连接设定阈值
    reduce_ratio: 相似度度量的衰减系数
    返回: 连接判定 True or False
    """
    # 提取局部掩膜
    mask1 = low_mask[low_bbox[0]:low_bbox[2], low_bbox[1]:low_bbox[3]]
    mask2 = high_mask[high_bbox[0]:high_bbox[2], high_bbox[1]:high_bbox[3]]
    # 得到相对质心坐标
    y_centroid_low, x_centroid_low = compute_centroid(mask1)
    y_centroid_high, x_centroid_high = compute_centroid(mask2)
    # 得到绝对质心坐标
    y_centroid_low_all = y_centroid_low + low_bbox[0]
    x_centroid_low_all = x_centroid_low + low_bbox[1]
    y_centroid_high_all = y_centroid_high + high_bbox[0]
    x_centroid_high_all = x_centroid_high + high_bbox[1]
    # 计算质心距离
    y_centroid_distance = abs(y_centroid_low_all - y_centroid_high_all)
    x_centroid_distance = abs(x_centroid_low_all - x_centroid_high_all)
    distence = (y_centroid_distance ** 2 + x_centroid_distance ** 2) / (
                (low_size[0] + high_size[0]) ** 2 + (low_size[1] + high_size[1]) ** 2)

    # 以重心为中心扩展至相同大小，需要扩展四次
    # 行前扩充
    if y_centroid_low > y_centroid_high:
        mask2 = np.pad(mask2, ((y_centroid_low - y_centroid_high, 0), (0, 0)), 'constant')
    elif y_centroid_low < y_centroid_high:
        mask1 = np.pad(mask1, ((y_centroid_high - y_centroid_low, 0), (0, 0)), 'constant')
    # 行后扩充，由于行前已对齐，所以比较mask的shape即可完成padding
    if mask1.shape[0] > mask2.shape[0]:
        mask2 = np.pad(mask2, ((0, mask1.shape[0] - mask2.shape[0]), (0, 0)), 'constant')
    elif mask1.shape[0] < mask2.shape[0]:
        mask1 = np.pad(mask1, ((0, mask2.shape[0] - mask1.shape[0]), (0, 0)), 'constant')
    # 列前扩充
    if x_centroid_low > x_centroid_high:
        mask2 = np.pad(mask2, ((0, 0), (x_centroid_low - x_centroid_high, 0)), 'constant')
    elif x_centroid_low < x_centroid_high:
        mask1 = np.pad(mask1, ((0, 0), (x_centroid_high - x_centroid_low, 0)), 'constant')
    # 列后扩充 
    if mask1.shape[1] > mask2.shape[1]:
        mask2 = np.pad(mask2, ((0, 0), (0, mask1.shape[1] - mask2.shape[1])), 'constant')
    elif mask1.shape[1] < mask2.shape[1]:
        mask1 = np.pad(mask1, ((0, 0), (0, mask2.shape[1] - mask1.shape[1])), 'constant')
    # 计算iou
    mask1 = np.reshape(mask1, (-1, 1)).astype(np.float32)
    mask2 = np.reshape(mask2, (-1, 1)).astype(np.float32)
    # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
    area1 = np.sum(mask1, axis=0)[0]
    area2 = np.sum(mask2, axis=0)[0]

    # intersections and union
    # 计算点积,即交集,得到结果(instances1, instances2)
    intersections = np.dot(mask1.T, mask2)[0][0]
    # 计算并集

    # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
    union = area1 + area2 - intersections
    # 得到结果(instances1, instances2)的iou矩阵
    overlaps = intersections * (1 - min(reduce_ratio * distence, 1)) / union
    if overlaps > thre:
        return True
    else:
        return False


def mask3_metrics(y_distance, x_distance, low_size, high_size, low_bbox, high_bbox, low_mask, high_mask, thre, reduce_ratio):
    """
    功能: 基于包围盒中心距离的插值掩膜IoU
    y_distance & x_distance: 包围盒中心y距离和x距离
    low_size & high_size: 掩膜1和掩膜2包围盒的尺寸
    low_bbox & high_bbox: 掩膜1和掩膜2包围盒坐标(y1, x1, y2, x2)
    low_mask & high_mask: 掩膜1和掩膜2在原图尺寸的掩膜信息
    thre: 连接设定阈值
    reduce_ratio: 相似度度量的衰减系数
    返回: 连接判定 True or False
    """
    distence = (y_distance ** 2 + x_distance ** 2) / (
                (low_size[0] + high_size[0]) ** 2 + (low_size[1] + high_size[1]) ** 2)
    # 提取局部掩膜
    mask1 = low_mask[low_bbox[0]:low_bbox[2], low_bbox[1]:low_bbox[3]]
    mask2 = high_mask[high_bbox[0]:high_bbox[2], high_bbox[1]:high_bbox[3]]
    # 插值至相同大小
    resize_height = max(low_size[0], high_size[0])
    resize_width = max(low_size[1], high_size[1])
    mask1 = cv2.resize(mask1.astype(np.uint8), dsize=(resize_height, resize_width), interpolation=cv2.INTER_NEAREST)
    mask2 = cv2.resize(mask2.astype(np.uint8), dsize=(resize_height, resize_width), interpolation=cv2.INTER_NEAREST)
    # 计算iou
    mask1 = np.reshape(mask1, (-1, 1)).astype(np.float32)
    mask2 = np.reshape(mask2, (-1, 1)).astype(np.float32)
    # 计算不同instance的mask面积,列求和，大小为(instance_num,)，一维
    area1 = np.sum(mask1, axis=0)[0]
    area2 = np.sum(mask2, axis=0)[0]

    # intersections and union
    # 计算点积,即交集,得到结果(instances1, instances2)
    intersections = np.dot(mask1.T, mask2)[0][0]
    # 计算并集

    # area1[:, None]为instances1行1列，area2[None, :]为1行instances2列，
    union = area1 + area2 - intersections
    # 得到结果(instances1, instances2)的iou矩阵
    overlaps = intersections * (1 - min(reduce_ratio * distence, 1)) / union
    if overlaps > thre:
        return True
    else:
        return False


def Hu_metrics(y_distance, x_distance, low_size, high_size, low_bbox, high_bbox, low_mask, high_mask, thre, reduce_ratio):
    """
    功能: 基于包围盒中心距离的Hu距离
    y_distance & x_distance: 包围盒中心y距离和x距离
    low_size & high_size: 掩膜1和掩膜2包围盒的尺寸
    low_bbox & high_bbox: 掩膜1和掩膜2包围盒坐标(y1, x1, y2, x2)
    low_mask & high_mask: 掩膜1和掩膜2在原图尺寸的掩膜信息
    thre: 连接设定阈值
    reduce_ratio: 相似度度量的衰减系数
    返回: 连接判定 True or False
    """
    distence = (y_distance ** 2 + x_distance ** 2) / (
                (low_size[0] + high_size[0]) ** 2 + (low_size[1] + high_size[1]) ** 2)
    # 提取局部掩膜
    mask1 = low_mask[low_bbox[0]:low_bbox[2], low_bbox[1]:low_bbox[3]]
    mask2 = high_mask[high_bbox[0]:high_bbox[2], high_bbox[1]:high_bbox[3]]
    # 向周围扩展至相同大小
    # 行扩充
    if low_size[0] > high_size[0]:
        # j_item_index，即mask2需要扩展
        height_pad = low_size[0] - high_size[0]
        pad_before = height_pad // 2
        pad_after = height_pad - pad_before
        # 默认填充False
        mask2 = np.pad(mask2, ((pad_before, pad_after), (0, 0)), 'constant')
    elif low_size[0] < high_size[0]:
        # i_item_index，即masks1需要扩展
        height_pad = high_size[0] - low_size[0]
        pad_before = height_pad // 2
        pad_after = height_pad - pad_before
        mask1 = np.pad(mask1, ((pad_before, pad_after), (0, 0)), 'constant')
    # 列扩充
    if low_size[1] > high_size[1]:
        # j_item_index，即masks2需要扩展
        width_pad = low_size[1] - high_size[1]
        pad_before = width_pad // 2
        pad_after = width_pad - pad_before
        mask2 = np.pad(mask2, ((0, 0), (pad_before, pad_after)), 'constant')
    elif low_size[1] < high_size[1]:
        # i_item_index，即masks1需要扩展
        width_pad = high_size[1] - low_size[1]
        pad_before = width_pad // 2
        pad_after = width_pad - pad_before
        mask1 = np.pad(mask1, ((0, 0), (pad_before, pad_after)), 'constant')
    # 转化数据
    Hu_mask1 = 255 * mask1.astype(np.uint8)
    Hu_mask2 = 255 * mask2.astype(np.uint8)
    Hu_mask1 = Hu_mask1[:, :, np.newaxis]
    Hu_mask2 = Hu_mask2[:, :, np.newaxis]

    contours1, _ = cv2.findContours(Hu_mask1, 2, 1)
    cnt1 = contours1[0]
    contours2, _ = cv2.findContours(Hu_mask2, 2, 1)
    cnt2 = contours2[0]
    # ret越小越接近
    ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
    # 限制大小
    ret = min(ret, 10)
    # 归一化
    ret = (10 - ret) / 10
    # 相似度度量
    overlaps = ret * (1 - min(reduce_ratio * distence, 1))
    if overlaps > thre:
        return True
    else:
        return False
#---------------------------------------connection functions------------------------------------------------
def connection(image, mode='mask0', threshold=[0.0], connectivity=2, max_instance_one_img=500,
                    start_layer=0, end_layer=0, box_expend=-1, **kwargs):
    """
    三维连接算法
    image: z,y,x, 二值三维标签数据, np.array
    mode: 相似性度量算法, str
        'box': bounding box IoU
        'mask0': mask IoU
        'mask1': mask IoU Center
        'mask2': mask IoU Centroid
        'mask3': mask IoU Interpolation
        'Hu': Hu Invariant Moment
    connectivity: 连接的层数，1代表只考虑邻层，2代表考虑邻层以及隔层
    max_instance_one_img: 单张图中最大的连通域(实例数量)
    start_layer：连接的起始层
    end_layer：连接的终止层
    box_expend: 包围盒预判定, 默认-1不进行判定, 否则两实例在扩展包围盒不重叠时直接选择不连接, 推荐1至2之间。

    返回: 连接并统一的三维数组
    """
    ####################################
    # 1.处理输入数据基本信息
    image_size = image.shape
    assert len(image_size) == 3
    assert mode in ['bbox', 'mask0', 'mask1', 'mask2', 'mask3', 'Hu']
    if mode != 'bbox' and mode != 'mask0':
        assert len(threshold) == 2
    assert connectivity >= 1
    ####################################
    # 2.初始化数据记录
    # 初始化instance计数器
    instance_num = 0
    # 记录instance标签值的字典
    instance_item = {}
    # 连接开始的层数
    start = start_layer
    # 连接结束的层数
    if end_layer == 0:
        end = image_size[0]
    # 记录每张图像的连通域实例信息
    # 键：'mask' 值：
    dataset_info={}
    # 由于分裂融合删除的多余的标签id，需要后续重复使用
    delet_set = []
    # 每张图设置的最多连通域数目
    max_instance_one_img = max_instance_one_img
    # 每个实例的连接记录  大小：层数*连通域实例数目  初始值-1
    construction_record = np.ones(shape=(image_size[0], max_instance_one_img), dtype=int) * -1
    ###################################
    # 3.开始连接
    for low_image_index in range(start, end-1):
        # 连接为两层的电镜进行连接，i为连接的底层，j为连接的上层，因此只需要到end-1即可
        if low_image_index not in dataset_info:
            dataset_info[low_image_index] = find_object(image[low_image_index], area_min=0, area_max=0)

        low_boxes = dataset_info[low_image_index]['bbox']
        low_masks = dataset_info[low_image_index]['mask']

        low_instance_num = low_boxes.shape[0]
        low_height = low_boxes[:, 2] - low_boxes[:, 0]
        low_width = low_boxes[:, 3] - low_boxes[:, 1]
        low_center_y = (low_boxes[:, 0] + 0.5 * low_height).astype(np.uint32)
        low_center_x = (low_boxes[:, 1] + 0.5 * low_width).astype(np.uint32)

        for high_image_add in range(connectivity):
            high_image_index = low_image_index + high_image_add + 1
            if high_image_index < end:
                if high_image_index not in dataset_info:
                    dataset_info[high_image_index] = find_object(image[high_image_index], area_min=0, area_max=0)
                high_boxes = dataset_info[high_image_index]['bbox']
                high_masks = dataset_info[high_image_index]['mask']

                high_instance_num = high_boxes.shape[0]
                high_height = high_boxes[:, 2] - high_boxes[:, 0]
                high_width = high_boxes[:, 3] - high_boxes[:, 1]
                high_center_y = (high_boxes[:, 0] + 0.5 * high_height).astype(np.int32)
                high_center_x = (high_boxes[:, 1] + 0.5 * high_width).astype(np.int32)

                # 开始计算，连接算法
                for i_item_index in range(low_instance_num):
                    for j_item_index in range(high_instance_num):
                        connection_decision = False

                        y_distance = abs(low_center_y[i_item_index] - high_center_y[j_item_index])
                        x_distance = abs(low_center_x[i_item_index] - high_center_x[j_item_index])

                        if box_expend != -1 and 2 * y_distance < box_expend * (low_height[i_item_index] + high_height[j_item_index]) \
                                and 2 * x_distance < box_expend * (low_width[i_item_index] + high_width[j_item_index]):
                            # 使用包围盒预判定，且两实例距离过远，直接判定不连接
                            pass
                        elif mode == 'bbox':
                            connection_decision = bbox_metrics(low_size=[low_height[i_item_index], low_width[i_item_index]], 
                            high_size=[high_height[j_item_index], high_width[j_item_index]], 
                            low_bbox=low_boxes[i_item_index], high_bbox=high_boxes[j_item_index], 
                            thre=threshold[0])
                        elif mode == 'mask0':
                            connection_decision = mask0_metrics(low_bbox=low_boxes[i_item_index], 
                            high_bbox=high_boxes[j_item_index], 
                            low_mask=low_masks[:, :, i_item_index], 
                            high_mask=high_masks[:, :, j_item_index], 
                            thre=threshold[0])
                        elif mode == 'mask1':
                            connection_decision = mask1_metrics(y_distance=y_distance, 
                            x_distance=x_distance, 
                            low_size=[low_height[i_item_index], low_width[i_item_index]], 
                            high_size=[high_height[j_item_index], high_width[j_item_index]], 
                            low_bbox=low_boxes[i_item_index], high_bbox=high_boxes[j_item_index], 
                            low_mask=low_masks[:, :, i_item_index], 
                            high_mask=high_masks[:, :, j_item_index], 
                            thre=threshold[0], 
                            reduce_ratio=threshold[1])
                        elif mode == 'mask2':
                            connection_decision = mask2_metrics(low_size=[low_height[i_item_index], low_width[i_item_index]], 
                            high_size=[high_height[j_item_index], high_width[j_item_index]], 
                            low_bbox=low_boxes[i_item_index], high_bbox=high_boxes[j_item_index], 
                            low_mask=low_masks[:, :, i_item_index], 
                            high_mask=high_masks[:, :, j_item_index], 
                            thre=threshold[0], 
                            reduce_ratio=threshold[1])
                        elif mode == 'mask3':
                            connection_decision = mask3_metrics(y_distance=y_distance, 
                            x_distance=x_distance, 
                            low_size=[low_height[i_item_index], low_width[i_item_index]], 
                            high_size=[high_height[j_item_index], high_width[j_item_index]], 
                            low_bbox=low_boxes[i_item_index], high_bbox=high_boxes[j_item_index], 
                            low_mask=low_masks[:, :, i_item_index], 
                            high_mask=high_masks[:, :, j_item_index], 
                            thre=threshold[0], 
                            reduce_ratio=threshold[1])
                        elif mode == 'Hu':
                            pass
                        else:
                            print(mode, ' function will coming soon!')
                        if connection_decision:
                            # 确认连接
                            print(low_image_index, '   layer   ', i_item_index, '   instance   ',
                                    construction_record[low_image_index, i_item_index], '<--->',
                                    high_image_index, '   layer   ', j_item_index, '   instance   ',
                                    construction_record[high_image_index, j_item_index])

                            if construction_record[low_image_index, i_item_index] == -1 and construction_record[
                                high_image_index, j_item_index] == -1:
                                # 这对连通域实例没有连接过
                                if len(delet_set) == 0:
                                    # 删除集合中没有元素
                                    # 标签计数器+1
                                    instance_num = instance_num + 1
                                    # 将两者的坐标放入字典中
                                    instance_item[instance_num] = [[low_image_index, i_item_index],
                                                                            [high_image_index, j_item_index]]
                                    # 更新链接状态记录矩阵
                                    construction_record[low_image_index, i_item_index] = instance_num
                                    construction_record[high_image_index, j_item_index] = instance_num
                                else:
                                    # 重用删除集合中的name，取出最后一个值
                                    instance_item[delet_set[-1]] = [[low_image_index, i_item_index],
                                                                            [high_image_index, j_item_index]]
                                    # 更新链接状态记录矩阵
                                    construction_record[low_image_index, i_item_index] = delet_set[-1]
                                    construction_record[high_image_index, j_item_index] = delet_set[-1]
                                    # 删除重用的集合元素
                                    del delet_set[-1]

                            elif construction_record[low_image_index, i_item_index] == -1 and construction_record[
                                high_image_index, j_item_index] != -1:
                                # 底层的突触对象没有连接过，上层的突出对象已连接
                                # 将未连接突触对象的放入对应的字典
                                instance_item[construction_record[high_image_index, j_item_index]].append(
                                    [low_image_index, i_item_index])
                                # 更新链接状态记录矩阵
                                construction_record[low_image_index, i_item_index] = \
                                    construction_record[high_image_index, j_item_index]
                            elif construction_record[low_image_index, i_item_index] != -1 and construction_record[
                                high_image_index, j_item_index] == -1:
                                # 上层的突触对象没有链接过，下层的突出对象已连接
                                # 将未连接突触对象的放入对应的字典
                                instance_item[construction_record[low_image_index, i_item_index]].append(
                                    [high_image_index, j_item_index])
                                # 更新链接状态记录矩阵
                                construction_record[high_image_index, j_item_index] = \
                                    construction_record[low_image_index, i_item_index]
                            else:
                                # 上下层对象都已链接过
                                if construction_record[low_image_index, i_item_index] == \
                                        construction_record[high_image_index, j_item_index]:
                                    # 上下层对象为同一个id，直接pass
                                    pass
                                else:
                                    # 上下层对象不一，进行合并、删除
                                    # 放弃上层的突触计数id，统一为下层的突触计数id
                                    delet_name = construction_record[high_image_index, j_item_index]
                                    delet_set.append(delet_name)
                                    # 根据字典的记录更改状态矩阵

                                    # 合并字典数据
                                    instance_item[construction_record[low_image_index, i_item_index]] = \
                                    instance_item[construction_record[low_image_index, i_item_index]] + \
                                    instance_item[delet_name]
                                    # 根据字典的记录更改状态矩阵
                                    for dict_delet in instance_item[delet_name]:
                                        construction_record[dict_delet[0], dict_delet[1]] = \
                                            construction_record[low_image_index, i_item_index]

                                    # 删除对应的键值对
                                    del instance_item[delet_name]

    ############################################################
    # 4.整理连接数据， 重复使用冗余删除的标签，舍弃较大的标签值
    while len(delet_set) != 0:

        if delet_set[-1] == instance_num:
            # 冗余标签刚好是最后一个，直接删除即可
            del delet_set[-1]
            instance_num = instance_num - 1
        else:
            # 重用删除集合中的name，取出最后一个值
            instance_item[delet_set[-1]] = copy.deepcopy(instance_item[instance_num])
            # 根据字典的记录更改状态矩阵
            for i in instance_item[delet_set[-1]]:
                construction_record[i[0], i[1]] = delet_set[-1]
            # 删除重用的集合元素
            del delet_set[-1]
            del instance_item[instance_num]
            instance_num = instance_num - 1

    ##########################################################
    # 5.根据construction_record和instance_item对原始数据进行统一赋值
    out_image = np.zeros(shape=image.shape, dtype=np.uint16)
    # 没有连接的实例连通域标签统一记为instance_num + 1
    instance_num += 1
    print('Unconnected label value:', instance_num)
    for construction_layer in range(image_size[0]):
        layer_mask = dataset_info[construction_layer]['mask'].copy().astype(np.uint16)
        for instance_index in range(layer_mask.shape[-1]):
            if construction_record[construction_layer, instance_index] == -1:
                out_image[construction_layer] += layer_mask[:, :, instance_index] * instance_num
            else:
                out_image[construction_layer] += layer_mask[:, :, instance_index] * construction_record[construction_layer, instance_index]

        print('complete layer ', construction_layer)
    # for construction_layer in range(image_size[0]):
    #     layer_mask = dataset_info[construction_layer]['mask'].copy().astype(np.uint16)
    #     add_layer_index = np.where(construction_record[construction_layer] != -1)[0]
    #     for instance_index in add_layer_index:
    #         out_image[construction_layer] += layer_mask[:, :, instance_index] * construction_record[construction_layer, instance_index]

    #     print('complete layer ', construction_layer)
    return out_image

if __name__ == '__main__':
    # 测试二值数据
    binary_volume = io.imread("Labels.tif")
    # 连接数据
    # Multi_volume = connection(binary_volume, mode='bbox', threshold=[0.025], connectivity=2, max_instance_one_img=500)
    # Multi_volume = connection(binary_volume, mode='mask0', threshold=[0.0], connectivity=2, max_instance_one_img=500)
    Multi_volume = connection(binary_volume, mode='mask1', threshold=[0.0, 8.0], connectivity=2, max_instance_one_img=500)
    # Multi_volume = connection(binary_volume, mode='mask2', threshold=[0.125, 10.0], connectivity=2, max_instance_one_img=500)
    # Multi_volume = connection(binary_volume, mode='mask3', threshold=[0.075, 6.8], connectivity=2, max_instance_one_img=500)
    # Hu 存在一点问题，参数需要调整
    # Multi_volume = connection(binary_volume, mode='Hu', threshold=[0.0, 0.0], connectivity=2, max_instance_one_img=500)
    # 保存
    io.imsave("Labels_connection.tif", Multi_volume)

    print('ok')