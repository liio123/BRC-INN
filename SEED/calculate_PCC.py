import numpy as np
from sklearn.decomposition import PCA
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 特征向量归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# PCA 将数据降维到C×C （C是通道）
def sby_dim_re(pcc):
    pca1 = PCA(n_components=1)
    pca2 = PCA(n_components=1)

    pcc_3 = pcc.reshape(pcc.shape[0], -1).transpose(1, 0)
    pca1.fit(pcc_3)
    pcc_3 = pca1.fit_transform(pcc_3)
    # print(pcc_3.shape)

    pcc_2 = pcc_3.reshape(pcc.shape[1], -1).transpose(1, 0)
    pca2.fit(pcc_2)
    pcc_2 = pca2.fit_transform(pcc_2)
    # print(pcc_2.shape)

    pcc_1 = pcc_2.reshape(pcc.shape[2], -1)
    # print(pcc_1.shape)

    return pcc_1

# 原始laplacian矩阵
def unnormalized_laplacian(adj_matrix):
    # 先求度矩阵
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix

DE_list = []
ScoreArousal_label_list = []
for i in range(1, 24):
    eeg = np.load(f"DE/{i}-DE-EEG.npy")    # (926, 5, 14, 8)
    ecg = np.load(f"DE/{i}-DE-ECG.npy")    # (926, 5, 2, 8)
    label = np.load(rf"D:\Program Files\JetBrains\project\XXX\label\\{i}-ScoreArousal-label.npy")

    eeg_pca = sby_dim_re(eeg)   # (14, 8)
    ecg_pca = sby_dim_re(ecg)   # (2, 8)

    # 计算皮尔逊相关系数
    rho = np.corrcoef(eeg_pca, ecg_pca) # (16, 16)
    metric_14 = rho[:14, :14]   # (14, 14)
    metric_2 = rho[14:17, 14:17]    # (2, 2)
    metric_14_2 = rho[:14, 14:17]   # (14, 2)

    # 归一化拉普拉斯特征值
    nor_lp_14 = unnormalized_laplacian(metric_14)
    lpls_eigenvalue_14, _ = np.linalg.eigh(nor_lp_14) # (14,)
    lpls_eigenvalue_14 = normalization(lpls_eigenvalue_14)

    nor_lp_2 = unnormalized_laplacian(metric_2)
    lpls_eigenvalue_2, _ = np.linalg.eigh(nor_lp_2) # (2,)
    lpls_eigenvalue_2 = normalization(lpls_eigenvalue_2)

    # 奇异值分解
    U, _, Vh = np.linalg.svd(metric_14_2)   # (14, 14), (2, 2)

    # 归一化拉普拉斯特征值
    U_lp = unnormalized_laplacian(U)
    lpls_U_eigVal_14, _ = np.linalg.eigh(U_lp) # (14,)
    lpls_U_eigVal_14 = normalization(lpls_U_eigVal_14)

    # 归一化拉普拉斯特征值
    Vh_lp = unnormalized_laplacian(Vh)
    lpls_Vh_eigVal_2, _ = np.linalg.eigh(Vh_lp) # (2,)
    lpls_Vh_eigVal_2 = normalization(lpls_Vh_eigVal_2)

    # de * weight
    eeg_weight = eeg.transpose(0, 1, 3, 2) * lpls_eigenvalue_14 # (926, 5, 8, 14)
    ecg_weight = ecg.transpose(0, 1, 3, 2) * lpls_eigenvalue_2  # (926, 5, 8, 2)

    # concatenate channel
    eeg_ecg = np.concatenate([eeg_weight * lpls_U_eigVal_14, ecg_weight * lpls_Vh_eigVal_2], axis=3)    # (926, 5, 8, 16)
    eeg_ecg = eeg_ecg.transpose(0, 3, 1, 2)     # (926, 16, 5, 8)
    print(eeg_ecg.shape)
    np.save(rf"D:\Program Files\JetBrains\project\XXX\weight_de\DE{i}.npy", eeg_ecg)
    DE_list.append(eeg_ecg)
    ScoreArousal_label_list.append(label)
DE = np.concatenate(DE_list)
labels = np.concatenate(ScoreArousal_label_list)
print(DE.shape)
print(labels.shape)
# np.save("D:\Program Files\JetBrains\project\zzh_project\DE.npy", DE)
np.save("D:\Program Files\JetBrains\project\XXX\ScoreArousal_label.npy", labels)


