import xarray as xr
import numpy as np
import os
#拆分整个研究区数据为8个子区域，以便后续处理，并对数据做归一化

low_res_path = 'D:\\pythonProject_DT\\WW3low'
hi_res_path = 'D:\\pythonProject_DT\\WW3hi'
# lon_min = 145.15
# lon_max = 147.65
# lat_min = -10.05
# lat_max = -7.55
# 8个子图得边界
bounds = [
    [135.2, 152.45, 0.3,-11.25 ],
    [135.2, 152.45, -11.15, -22.45],
    [135.2, 152.45, -22.35, -33.65],
    [135.2, 152.45, -33.55, -44.85],
    [152.35, 170.1, 0.3, -11.25],
    [152.35, 170.1, -11.15, -22.45],
    [152.35, 170.1, -22.35, -33.65],
    [152.35, 170.1, -33.55, -44.85]
]
#训练集、验证集、测试集数据划分比例
split_ratio = [0.4, 0.2, 0.4]
variable_name=['hs','t02','tm0m1','U10','V10','dir']
for i in range(8):
    for var in variable_name:
        data, hsZ_data, Hi_data, = process_data(low_res_path, hi_res_path, bounds[i][0], bounds[i][1], bounds[i][3], bounds[i][2],var)

        folder_path1 = 'D:\\pythonProject_DT\\'+str(i + 1)+'\\OCdata\\'
        folder_path2 = 'D:\\pythonProject_DT\\'+str(i + 1)+'\\OZCdata\\'
        folder_path3 = 'D:\\pythonProject_DT\\'+str(i + 1)+'\\OWW3Cdata\\'
        data_split, hs_file_names = split_and_save_data(data, split_ratio=split_ratio, prefix=var + '_data',folder=folder_path1)
        hsZ_data_split, hsZ_file_names = split_and_save_data(hsZ_data, split_ratio=split_ratio, prefix=var + 'Z_data',folder=folder_path2)
        Hi_data_split, Hi_hs_file_names = split_and_save_data(Hi_data, split_ratio=split_ratio,prefix='Hi_' + var + '_data',folder=folder_path3)


def Zscore(feature, mean, std):
    """
    使用给定的均值和标准差对数据进行归一化。

    参数：
    - feature: 需要归一化的数据，形状为 (样本数, ...)
    - mean: 训练集的均值，形状应与 feature 的最后一个维度匹配
    - std: 训练集的标准差，形状应与 feature 的最后一个维度匹配

    返回：
    - feature_normalized: 归一化后的数据
    """
    # 避免除以零的情况
    std = np.where(std == 0, 1, std)  # 如果标准差为0，则将其设为1

    # 使用给定的均值和标准差进行归一化
    feature_normalized = (feature - mean) / std
    return np.nan_to_num(feature_normalized, nan=0)
    
def process_data(low_res_path, hi_res_path, lon_min, lon_max, lat_min, lat_max, variable_name):
    filelow_list = os.listdir(low_res_path)
    filelow_list.sort()
    filehi_list = os.listdir(hi_res_path)
    filehi_list.sort()

    # 初始化数据列表
    dataL = []
    dataH = []

    # 读取低分辨率数据
    for file_name in filelow_list:
        if file_name.endswith('.nc4'):
            file_path = os.path.join(low_res_path, file_name)
            dsL = xr.open_dataset(file_path)
            dsL_subset = dsL.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
            timeL = dsL_subset['time'].values
            hsL = dsL_subset[variable_name].values
            for i in range(0, len(timeL), 6):  # 每隔6个数据取一个
                dataL.append([hsL[i]])
            dsL.close()

    # 读取高分辨率数据
    for file_name in filehi_list:
        if file_name.endswith('.nc4'):
            file_path = os.path.join(hi_res_path, file_name)
            dsH = xr.open_dataset(file_path)
            dsH_subset = dsH.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
            timeH = dsH_subset['time'].values
            hsH = dsH_subset[variable_name].values
            for i in range(0, len(timeH), 6):  # 每隔6个数据取一个
                dataH.append([hsH[i]])
            dsH.close()

    # 转换为三维数组
    data = np.array(dataL)
    # 去掉第二维度
    data = np.squeeze(data, axis=1)
    # 假设 dataL 是二维列表，提取值
    Hi_data = np.array(dataH)
    Hi_data=np.squeeze(Hi_data, axis=1)
    # 提取训练集数据（前40%）
    train_size = int(len(data) * split_ratio[0])
    train_data = data[:train_size]

    # 计算训练集的均值和标准差
    train_mean = np.nanmean(train_data)
    train_std = np.nanstd(train_data)

    # 对整个数据集进行归一化
    dataLZ = Zscore(data, train_mean, train_std)
     # 高分辨率数据也使用相同的均值和标准差

    return data, dataLZ, Hi_data
def split_and_save_data(data, split_ratio, prefix='',folder=''):
    total_len = len(data)
    split_points = [int(total_len * ratio) for ratio in split_ratio]

    split_data = []
    start_idx = 0
    for split_point in split_points:
        split_data.append(data[start_idx:start_idx + split_point])
        start_idx += split_point

    file_names = [prefix + '_train.npy', prefix + '_valid.npy', prefix + '_test.npy']
    for i in range(len(split_data)):
        # np.save(file_names[i],split_data[i])
        np.save(os.path.join(folder,file_names[i]),split_data[i])

    return split_data, file_names

