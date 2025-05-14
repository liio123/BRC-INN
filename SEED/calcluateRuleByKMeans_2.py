import scipy.io as sio
import os
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

np.set_printoptions(threshold=np.inf)
# 被试
subjects = [
            '1_20131027.mat.npy', '1_20131030.mat.npy', '1_20131107.mat.npy', '2_20140404.mat.npy', '2_20140413.mat.npy', '2_20140419.mat.npy',
            '3_20140603.mat.npy', '3_20140611.mat.npy', '3_20140629.mat.npy', '4_20140621.mat.npy', '4_20140702.mat.npy', '4_20140705.mat.npy',
            '5_20140411.mat.npy', '5_20140418.mat.npy', '5_20140506.mat.npy', '6_20130712.mat.npy', '6_20131016.mat.npy', '6_20131113.mat.npy',
            '7_20131027.mat.npy', '7_20131030.mat.npy', '7_20131106.mat.npy', '8_20140511.mat.npy', '8_20140514.mat.npy', '8_20140521.mat.npy',
            '9_20140620.mat.npy', '9_20140627.mat.npy', '9_20140704.mat.npy',
            '10_20131130.mat.npy', '10_20131204.mat.npy', '10_20131211.mat.npy', '11_20140618.mat.npy', '11_20140625.mat.npy',
            '11_20140630.mat.npy',
            '12_20131127.mat.npy', '12_20131201.mat.npy', '12_20131207.mat.npy', '13_20140527.mat.npy', '13_20140603.mat.npy',
            '13_20140610.mat.npy',
            '14_20140601.mat.npy', '14_20140615.mat.npy', '14_20140627.mat.npy', '15_20130709.mat.npy', '15_20131016.mat.npy',
            '15_20131105.mat.npy',
            ]

# 超参数
# 通道名顺序
ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
            'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
            'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
            'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

# 采样频率
# sfreq = 200
# 每个.mat文件中的数据label
# basic_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
# basic_label = [label + 1 for label in basic_label]


# def read_one_file(file_path):
#     """
#     input:单个.mat文件路径
#     output:raw格式数据
#     """
#
#     data = sio.loadmat(file_path)
#
#     # 获取keys并转化为list，获取数据所在key
#     keys = list(data.keys())[3:]
#     # print(keys)
#     # 获取数据
#     train_data = np.empty([0, 62])
#     label = []
#     for i in range(len(keys)):
#         # 获取数据
#         stamp = data[keys[i]][:, 1:].transpose([1, 0])
#         # print(stamp.shape)
#         train_data = np.vstack([train_data, stamp])
#
#         label.extend(np.repeat(basic_label[i], stamp.shape[0] / sfreq))
#     train_data = train_data.reshape(-1, 200, 62).transpose(0, 2, 1)
#     print(train_data.shape)
#     # print(label)
#     print(len(label))
#     return train_data, label


# def find_indexes(array):
#     index_dict = {}
#     for i, num in enumerate(array):
#         if num not in index_dict:
#             index_dict[num] = [i]
#         else:
#             index_dict[num].append(i)
#     return index_dict


def is_one2one(list1, list2):
    # 计算相关性系数
    kv = {}
    for i in range(62):
        if list1[i] not in kv:
            kv[list1[i]] = list2[i]
        else:
            if kv[list1[i]] != list2[i]:
                return False
    return True


def kmeans_(train_data, labelss, op):
    rules = {}
    label = np.zeros((len(train_data), 3))
    label_all = []
    cnt = 0
    # for i in range(5):
    for i in tqdm(range(len(train_data))):
        X = train_data[i]
        y = int(labelss[i])
        kmeans = KMeans(n_clusters=op)
        results = kmeans.fit_predict(X)
        # 处理分类效果
        results = results.tolist()

        # 计算映射
        # flag 判断是否出现
        flag = False
        for key, rule in rules.items():
            if is_one2one(results, rule):
                label[key][y] = 1 + label[key][y]
                flag = True
        if flag == False:
            rules[cnt] = results
            label[cnt][y] = 1
            cnt += 1
        pass

        # index_dict = find_indexes(results)
        # print(index_dict)
    # 当前簇最优解及其分布
    max_index = np.argmax(label.sum(axis=1))
    return rules[max_index], label[max_index]


if __name__ == '__main__':
    dir = "G:\XXX\SEED\DE"

    for subject in subjects:
        # subject = "1_20131027.mat.npy"
        file_path = os.path.join(dir, subject)
        train_data = np.load(file_path)
        print(train_data.shape)
        # (3394, 62, 200) 3394
        # 读取数据
        # train_data, label = read_one_file(file_path)
        label = np.concatenate(
            [np.full((235, 1), 2), np.ones([233, 1]), np.zeros([206, 1]), np.zeros([238, 1]), np.ones([185, 1]),
             np.full((195, 1), 2), np.zeros([237, 1]),
             np.ones([216, 1]), np.full((265, 1), 2), np.full((237, 1), 2), np.ones([235, 1]), np.zeros([233, 1]),
             np.ones([235, 1]), np.full((238, 1), 2),
             np.ones([206, 1])])
        label = np.squeeze(label)
        # 先验规则
        rule_list = {}
        label_list = {}
        for i in range(2, 6):
            print("op", i)
            rules_i, label_i = kmeans_(train_data, label, op=i)
            rule_list[i] = rules_i
            label_list[i] = label_i
            print(rules_i)
            print(label_i)

        # 记录每种规则的情况
        df = pd.DataFrame(rule_list)
        # 将DataFrame写入Excel文件
        df.to_excel(f"prob_3\{subject.split('.')[0]}.xlsx", index=False)
        df = pd.DataFrame(label_list)
        # 将DataFrame写入Excel文件
        df.to_excel(f"prob_3\{subject.split('.')[0]}_rules.xlsx", index=False)
        # 计算概率
        for l in range(2, 6):
            print(label_list[l])
            total = label_list[l][0] + label_list[l][1] + label_list[l][2]
            if total == 0:
                continue
            for j, ll in enumerate(label_list[l]):
                label_list[l][j] = ll / total

        # 提取字典的值并转换为列表
        values_list = list(label_list.values())
        # 将列表转换为Numpy数组
        numpy_array = np.array(values_list)
        np.save(f"prob_3\{subject.split('.')[0]}_rules.npy", numpy_array)

# 使用
# x = np.load('./prob_3/1_20131027_rules.npy')
# print(x)
# print(x[0])
