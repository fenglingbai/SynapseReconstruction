# connection log
import os
import pickle
import joblib
import numpy as np
import cv2

root_path = "fusion_outcome_v1"
# save_data = {}
#
# save_data["img"] = all_img
# save_data["gt_bbox"] = whole_bbox
# save_data["gt_mask"] = whole_mask
# save_data["gt_class_id"] = whole_class_ids
# # save_data["gt_slice_id"] = temp_slice_id
# save_data["class_names"] = dataset_train.class_names
#
# save_data["pre_bbox"] = temp_save_bbox
# save_data["pre_mask"] = temp_save_mask
# save_data["pre_class_id"] = temp_save_class_id
# save_data["pre_score"] = temp_save_scores
# save_data["pre_slice_id"] = temp_save_slice_id
save_path = "constrution_outcome_v3"
data_list = os.listdir(root_path)

data_list.sort(key=lambda x: int(x[:-7]))
# 初始化synapse计数器
synapse_num = 0
# 记录synapse的字典
synapse_item = {}
# 上下层链接的层数
construction_layer = 2
# 链接的mask阈值
threshold = 0.125
# box扩展系数
box_expend = 1.2
# 距离衰减系数
reduce_ratio = 10
# 删除的需要重复使用的id
delet_set = []
# construction_record = []
# 一张图中最多的instance实例个数
max_instance_one_img = 200
# 重建开始的层数
start = 0
# 重建结束的层数
end = len(data_list)
# 子区域高的切割数量
high_length = 5
# 子区域宽的切割数量
width_length = 5

construction_record = np.ones(shape=(len(data_list), max_instance_one_img), dtype=int) * -1

assert construction_layer >= 1
# 连接为两层的电镜进行连接，i为连接的底层，j为连接的上层，因此只需要到end-1即可
for low_image in range(start, end-1):
    low_data_item = data_list[low_image]
    low_data_item = os.path.join(root_path, low_data_item)
    with open(low_data_item, 'rb') as f_in:
        data = joblib.load(f_in)
        print('low_data has been read from {}'.format(low_data_item))
    low_boxes = data['pre_bbox']
    low_mask = data['pre_mask']
    low_slice_id = data['pre_slice_id']

    low_instance_num = low_boxes.shape[0]
    low_height = low_boxes[:, 2] - low_boxes[:, 0]
    low_width = low_boxes[:, 3] - low_boxes[:, 1]
    low_center_y = (low_boxes[:, 0] + 0.5 * low_height).astype(np.uint32)
    low_center_x = (low_boxes[:, 1] + 0.5 * low_width).astype(np.uint32)


    for high_image in range(construction_layer):
        if low_image+high_image+1 >= end:
            pass
        else:
            high_data_item = data_list[low_image+high_image+1]
            high_data_item = os.path.join(root_path, high_data_item)
            with open(high_data_item, 'rb') as f_in:
                data = joblib.load(f_in)
                print('heigh_data has been read from {}'.format(high_data_item))

            high_boxes = data['pre_bbox']
            high_mask = data['pre_mask']
            high_slice_id = data['pre_slice_id']

            high_instance_num = high_boxes.shape[0]
            high_height = high_boxes[:, 2] - high_boxes[:, 0]
            high_width = high_boxes[:, 3] - high_boxes[:, 1]
            high_center_y = (high_boxes[:, 0] + 0.5 * high_height).astype(np.int32)
            high_center_x = (high_boxes[:, 1] + 0.5 * high_width).astype(np.int32)

            low_slice_index = []
            high_slice_index = []
            # 遍历所有的区域，找出区域的起始突触的Index
            for i in range(high_length * width_length):
                # 计算low_slice_index
                if len(np.where(low_slice_id == i)[0]) == 0:
                    # 该区域没有突触实例
                    if len(low_slice_index) != 0:
                        # 分块列表存在数据
                        low_slice_index.append(low_slice_index[-1])
                    else:
                        # 分块列表不存在数据，一直填充0
                        low_slice_index.append(0)
                else:
                    low_slice_index.append(np.where(low_slice_id == i)[0][0])

                # 计算high_slice_index
                if len(np.where(high_slice_id == i)[0]) == 0:
                    # 该区域没有突触实例
                    if len(high_slice_index) != 0:
                        # 分块列表存在数据
                        high_slice_index.append(high_slice_index[-1])
                    else:
                        # 分块列表不存在数据，一直填充0
                        high_slice_index.append(0)
                else:
                    high_slice_index.append(np.where(high_slice_id == i)[0][0])
            # 加入中止标志
            low_slice_index.append(low_boxes.shape[0])
            high_slice_index.append(high_boxes.shape[0])

            for i in range(high_length * width_length):
                # 遍历不同的子区域，进行局部融合
                # 下层子区域的所有索引
                compute_self_index = [index for index in range(low_slice_index[i], low_slice_index[i + 1])]

                # 上层子区域的所有索引
                # divmod(a,b): (a // b, a % b)
                def nearest_slice(id, hight, width, length):
                    row, col = divmod(id, width)

                    row_array = np.array([i for i in range(row - length, row + length + 1)])
                    row_array = row_array[(row_array >= 0) * (row_array < hight)]

                    col_array = np.array([i for i in range(col - length, col + length + 1)])
                    col_array = col_array[(col_array >= 0) * (col_array < width)]

                    row_array, col_array = np.meshgrid(col_array, row_array)

                    index = np.stack((col_array, row_array), axis=2)
                    index = np.reshape(index, (-1, 2))

                    nearest_index = [item[0] * width + item[1] for item in index]

                    return nearest_index


                compute_other_index = []
                slice_set = nearest_slice(i, high_length, width_length, 1)
                for slice_item in slice_set:
                    compute_other_index = compute_other_index + \
                                          [i for i in range(high_slice_index[slice_item],
                                                            high_slice_index[slice_item + 1])]



                # 开始计算，连接算法
                for i_item_index in compute_self_index:
                    for j_item_index in compute_other_index:
                        y_distance = abs(low_center_y[i_item_index] - high_center_y[j_item_index])
                        x_distance = abs(low_center_x[i_item_index] - high_center_x[j_item_index])
                        # # 找出实际连接情况
                        # gt_connection = (low_value[i_item_index]==high_value[j_item_index])
                        # # 对预测连接情况进行初始化
                        # pre_connection = False
                        if 2 * y_distance < box_expend * (low_height[i_item_index] + high_height[j_item_index]) \
                                and 2 * x_distance < box_expend * (low_width[i_item_index] + high_width[j_item_index]):

                            # 提取局部掩膜
                            masks1 = low_mask[low_boxes[i_item_index, 0]:low_boxes[i_item_index, 2],
                                     low_boxes[i_item_index, 1]:low_boxes[i_item_index, 3],
                                     i_item_index]

                            masks2 = high_mask[high_boxes[j_item_index, 0]:high_boxes[j_item_index, 2],
                                     high_boxes[j_item_index, 1]:high_boxes[j_item_index, 3],
                                     j_item_index]

                            # 计算重心

                            # 计算low重心
                            weight_all_low = np.sum(masks1)

                            # y坐标
                            # low_weight_y 相对坐标
                            # all_low_weight_y 绝对坐标
                            area_count = 0
                            low_weight_y = 0
                            while (area_count < weight_all_low):
                                area_count = area_count + 2 * np.sum(masks1[low_weight_y, :])
                                low_weight_y = low_weight_y + 1

                            all_low_weight_y = low_weight_y + low_boxes[i_item_index, 0]

                            # x坐标
                            # low_weight_x 相对坐标
                            # all_low_weight_x 绝对坐标
                            area_count = 0
                            low_weight_x = 0
                            while (area_count < weight_all_low):
                                area_count = area_count + 2 * np.sum(masks1[:, low_weight_x])
                                low_weight_x = low_weight_x + 1

                            all_low_weight_x = low_weight_x + low_boxes[i_item_index, 1]

                            # 计算high重心
                            weight_all_high = np.sum(masks2)
                            # y坐标
                            # high_weight_y 相对坐标
                            # all_high_weight_y 绝对坐标
                            area_count = 0
                            high_weight_y = 0
                            while (area_count < weight_all_high):
                                area_count = area_count + 2 * np.sum(masks2[high_weight_y, :])
                                high_weight_y = high_weight_y + 1

                            all_high_weight_y = high_weight_y + high_boxes[j_item_index, 0]

                            # x坐标
                            # high_weight_x 相对坐标
                            # all_high_weight_x 绝对坐标
                            area_count = 0
                            high_weight_x = 0
                            while (area_count < weight_all_high):
                                area_count = area_count + 2 * np.sum(masks2[:, high_weight_x])
                                high_weight_x = high_weight_x + 1

                            all_high_weight_x = high_weight_x + high_boxes[j_item_index, 1]

                            y_weight_distance = abs(all_low_weight_y - all_high_weight_y)
                            x_weight_distance = abs(all_low_weight_x - all_high_weight_x)

                            y_distance = abs(low_center_y[i_item_index] - high_center_y[j_item_index])
                            x_distance = abs(low_center_x[i_item_index] - high_center_x[j_item_index])

                            distence = (y_weight_distance * x_weight_distance + 1) / (
                                        (low_height[i_item_index] + high_height[j_item_index]) * (
                                        low_width[i_item_index] + high_width[j_item_index]))

                            if low_weight_y > high_weight_y:
                                pad = low_weight_y - high_weight_y
                                # 默认填充False
                                # https://blog.csdn.net/zenghaitao0128/article/details/78713663
                                masks2 = np.pad(masks2, ((pad, 0), (0, 0)), 'constant')
                            # 行前扩充，low需要扩充
                            elif low_weight_y < high_weight_y:
                                # i_item_index，即masks1需要扩展
                                pad = high_weight_y - low_weight_y
                                masks1 = np.pad(masks1, ((pad, 0), (0, 0)), 'constant')

                            # 由于左上角顶点已经对齐，所以比较mask的shape，即可完成相关pad
                            # 行后扩充，high需要扩充
                            if masks1.shape[0] > masks2.shape[0]:
                                pad = masks1.shape[0] - masks2.shape[0]
                                # 默认填充False
                                # https://blog.csdn.net/zenghaitao0128/article/details/78713663
                                masks2 = np.pad(masks2, ((0, pad), (0, 0)), 'constant')
                            # 行后扩充，low需要扩充
                            elif masks1.shape[0] < masks2.shape[0]:
                                pad = masks2.shape[0] - masks1.shape[0]
                                # i_item_index，即masks1需要扩展
                                masks1 = np.pad(masks1, ((0, pad), (0, 0)), 'constant')

                            # 列前扩充，high需要扩充
                            if low_weight_x > high_weight_x:
                                pad = low_weight_x - high_weight_x
                                # 默认填充False
                                # https://blog.csdn.net/zenghaitao0128/article/details/78713663
                                masks2 = np.pad(masks2, ((0, 0), (pad, 0)), 'constant')

                            # 列前扩充，low需要扩充
                            elif low_weight_x < high_weight_x:
                                # i_item_index，即masks1需要扩展
                                pad = high_weight_x - low_weight_x
                                masks1 = np.pad(masks1, ((0, 0), (high_weight_x - low_weight_x, 0)), 'constant')

                            # 行后扩充，high需要扩充
                            if masks1.shape[1] > masks2.shape[1]:
                                pad = masks1.shape[1] - masks2.shape[1]
                                # 默认填充False
                                # https://blog.csdn.net/zenghaitao0128/article/details/78713663
                                masks2 = np.pad(masks2, ((0, 0), (0, pad)), 'constant')

                            # 行后扩充，low需要扩充
                            elif masks1.shape[1] < masks2.shape[1]:
                                pad = masks2.shape[1] - masks1.shape[1]
                                # i_item_index，即masks1需要扩展
                                masks1 = np.pad(masks1, ((0, 0), (0, pad)), 'constant')

                            # 计算iou
                            masks1 = np.reshape(masks1, (-1, 1)).astype(np.float32)
                            masks2 = np.reshape(masks2, (-1, 1)).astype(np.float32)
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
                            overlaps = intersections * (1 - min(reduce_ratio * distence, 1)) / union

                            if overlaps > threshold:
                                # 开始连接
                                print(low_image, 'image', i_item_index, 'synapse',
                                      construction_record[low_image, i_item_index], '<--->',
                                      low_image+high_image+1, 'image', j_item_index, 'synapse',
                                      construction_record[low_image+high_image+1, j_item_index])


                                if construction_record[low_image, i_item_index] == -1 and construction_record[
                                    low_image+high_image+1, j_item_index] == -1:
                                    # 这对突触没有链接过
                                    if len(delet_set) == 0:
                                        # 删除集合中没有元素
                                        # 计数器+1
                                        synapse_num = synapse_num + 1
                                        # 将两者的坐标放入字典中
                                        synapse_item["%04d" % synapse_num] = [[low_image, i_item_index],
                                                                              [low_image+high_image+1, j_item_index]]
                                        # 更新链接状态记录矩阵
                                        construction_record[low_image, i_item_index] = synapse_num
                                        construction_record[low_image+high_image+1, j_item_index] = synapse_num
                                    else:
                                        # 重用删除集合中的name，取出最后一个值
                                        synapse_item["%04d" % delet_set[-1]] = [[low_image, i_item_index],
                                                                                [low_image+high_image+1, j_item_index]]
                                        # 更新链接状态记录矩阵
                                        construction_record[low_image, i_item_index] = delet_set[-1]
                                        construction_record[low_image+high_image+1, j_item_index] = delet_set[-1]
                                        # 删除重用的集合元素
                                        del delet_set[-1]

                                elif construction_record[low_image, i_item_index] == -1 and construction_record[
                                    low_image+high_image+1, j_item_index] != -1:
                                    # 底层的突触对象没有链接过，上层的突出对象已连接
                                    # 将未连接突触对象的放入对应的字典
                                    synapse_item["%04d" % construction_record[low_image+high_image+1, j_item_index]].append(
                                        [low_image, i_item_index])
                                    # 更新链接状态记录矩阵
                                    construction_record[low_image, i_item_index] = \
                                        construction_record[low_image+high_image+1, j_item_index]
                                elif construction_record[low_image, i_item_index] != -1 and construction_record[
                                    low_image+high_image+1, j_item_index] == -1:
                                    # 上层的突触对象没有链接过，下层的突出对象已连接
                                    # 将未连接突触对象的放入对应的字典
                                    synapse_item["%04d" % construction_record[low_image, i_item_index]].append(
                                        [low_image+high_image+1, j_item_index])
                                    # 更新链接状态记录矩阵
                                    construction_record[low_image+high_image+1, j_item_index] = \
                                        construction_record[low_image, i_item_index]
                                else:
                                    # 上下层对象都已链接过
                                    if construction_record[low_image, i_item_index] == \
                                            construction_record[low_image+high_image+1, j_item_index]:
                                        # 上下层对象为同一个id，直接pass
                                        pass
                                    else:
                                        # 上下层对象不一，进行合并、删除
                                        # 放弃上层的突触计数id，统一为下层的突触计数id
                                        delet_name = construction_record[low_image+high_image+1, j_item_index]
                                        delet_set.append(delet_name)
                                        # 根据字典的记录更改状态矩阵

                                        # 合并字典数据
                                        synapse_item["%04d" % construction_record[low_image, i_item_index]] = \
                                        synapse_item["%04d" % construction_record[low_image, i_item_index]] + \
                                        synapse_item["%04d" % delet_name]
                                        # 根据字典的记录更改状态矩阵
                                        for dict_delet in synapse_item["%04d" % delet_name]:
                                            construction_record[dict_delet[0], dict_delet[1]] = \
                                                construction_record[low_image, i_item_index]

                                        # 删除对应的键值对
                                        del synapse_item["%04d" % delet_name]


while len(delet_set) != 0:
    # 重用删除集合中的name，取出最后一个值
    # synapse_num
    synapse_item["%04d" % delet_set[-1]] = synapse_item["%04d" % synapse_num]
    # 根据字典的记录更改状态矩阵
    for i in synapse_item["%04d" % delet_set[-1]]:
        construction_record[i[0], i[1]] = delet_set[-1]
    # 删除重用的集合元素
    del delet_set[-1]
    del synapse_item["%04d" % synapse_num]
    synapse_num = synapse_num-1

save_data = {}

save_data["construction_record"] = construction_record
save_data["synapse_item"] = synapse_item

# save
os.makedirs(save_path, exist_ok=True)
cur_outcome_path = os.path.join(save_path, "constrution_outcome.pickle")
with open(cur_outcome_path, 'wb') as f_out:
    joblib.dump(save_data, f_out)
    print(
        'save_data has been written to {}, and can be loaded when testing to ensure correct results'.format(
            cur_outcome_path))

print("end")