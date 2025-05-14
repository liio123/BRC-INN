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
subjects = ['1.npy', '2.npy', '3.npy', '4.npy', '5.npy', '6.npy', '7.npy', '8.npy', '9.npy', '10.npy', '11.npy', '12.npy', '13.npy', '14.npy', '15.npy']

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
    label = np.zeros((len(train_data), 4))
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
    dir = "G:/组合脑相关内容/SEED-IV/test/DE/3"

    for subject in subjects:
        # subject = "1_20131027.mat.npy"
        file_path = os.path.join(dir, subject)
        train_data = np.load(file_path)
        print(train_data.shape)
        # (3394, 62, 200) 3394
        # 读取数据
        # train_data, label = read_one_file(file_path)
        label = np.concatenate([np.ones([42, 1]), np.full((32,1), 2), np.full((23,1), 2), np.ones([45, 1]), np.full((48,1), 3), np.full((26,1), 3), np.full((64,1), 3), np.ones([23, 1]),
                          np.ones([26, 1]), np.full((16,1), 2), np.ones([51, 1]), np.zeros([41, 1]), np.full((39,1), 2), np.full((19,1), 3), np.full((28,1), 3), np.zeros([44, 1]),
                          np.full((14,1), 2), np.full((17,1), 3), np.zeros([45, 1]), np.zeros([22, 1]), np.full((39,1), 2), np.zeros([38, 1]), np.ones([41, 1]), np.zeros([39, 1])])
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
        df.to_excel(f"D:\Program Files\JetBrains\PyCharm Community Edition 2019.2.4\Ruleset\prob_3/{subject.split('.')[0]}.xlsx", index=False)
        df = pd.DataFrame(label_list)
        # 将DataFrame写入Excel文件
        df.to_excel(f"D:\Program Files\JetBrains\PyCharm Community Edition 2019.2.4\Ruleset\prob_3/{subject.split('.')[0]}_rules.xlsx", index=False)
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
        np.save(f"D:\Program Files\JetBrains\PyCharm Community Edition 2019.2.4\Ruleset\prob_3\{subject.split('.')[0]}_rules.npy", numpy_array)

# 使用
# x = np.load('./prob_3/1_20131027_rules.npy')
# print(x)
# print(x[0])
