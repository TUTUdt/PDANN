#
# import xarray as xr
# import numpy as np
# import os
# def Zscore(feature):
#     # 分别对每列数据进行归一化
#     feature_normalized = np.zeros_like(feature)  # 创建一个与原数组大小相同的归一化结果数组
#
#     for i in range(feature.shape[1]):  # 对每个列进行迭代
#         for j in range(feature.shape[2]):  # 对每个位置进行迭代
#             column_data = feature[:, i, j]  # 获取当前列数据
#
#             # 计算均值和标准差
#             mean = np.mean(column_data)
#             std = np.std(column_data)
#             # 使用Z-score归一化
#             feature_normalized[:, i, j] = (column_data - mean) / std
#             feature_normalized = np.nan_to_num(feature_normalized, nan=0)
#     return feature_normalized
# def process_data(low_res_path, hi_res_path, lon_min,lon_max,lat_min,lat_max):
#     filelow_list = os.listdir(low_res_path)
#     filelow_list.sort()
#     filehi_list = os.listdir(hi_res_path)
#     filehi_list.sort()
#     # # 指定经纬度范围
#     # lon_min = 135.2
#     # lon_max = 170
#     # lat_min = -50
#     # lat_max = 0.3
#     #整张图的某个矩形区域
#     # lon_min = 135.2
#     # lon_max = 149.25
#     # lat_min = -10.8
#     # lat_max = 0.3
#     # 初始化数据列表
#     dataL = []
#     dataH = []
#     # 读取.nc文件并保存数据
#     for file_name in filelow_list:
#         if file_name.endswith('.nc4'):
#             file_path = os.path.join(low_res_path, file_name)
#             # 裁剪数据
#             dsL = xr.open_dataset(file_path)
#             dsL_subset = dsL.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
#             # 提取数据并存入数据列表
#             timeL = dsL_subset['time'].values
#             hsL = dsL_subset['hs'].values
#             dirL_data = dsL_subset['dir'].values
#             t02L = dsL_subset['t02'].values
#             tm0m1L = dsL_subset['tm0m1'].values
#             U10L = dsL_subset['U10'].values
#             V10L = dsL_subset['V10'].values
#
#             for i in range(0, len(timeL), 6):
#                 dataL.append([hsL[i], dirL_data[i], t02L[i], tm0m1L[i], U10L[i], V10L[i]])
#
#             dsL.close()
#     del dsL, dsL_subset, timeL, hsL, dirL_data, t02L, tm0m1L, U10L, V10L
#
#     # 读取.nc文件并保存数据
#     for file_name in filehi_list:
#         if file_name.endswith('.nc4'):
#             file_path = os.path.join(hi_res_path, file_name)
#             # 裁剪数据
#             dsH = xr.open_dataset(file_path)
#             dsH_subset = dsH.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
#             # 提取数据并存入数据列表
#             timeH = dsH_subset['time'].values
#             hsH = dsH_subset['hs'].values#hs
#             dirH_data = dsH_subset['dir'].values
#             t02H = dsH_subset['t02'].values
#             tm0m1H = dsH_subset['tm0m1'].values
#             # U10H = dsH_subset['U10'].values
#             # V10H = dsH_subset['V10'].values
#
#             for i in range(0,len(timeH),6):
#                 dataH.append([hsH[i], dirH_data[i], t02H[i], tm0m1H[i],]) #U10H[i], V10H[i]
#             dsH.close()
#     del dsH_subset, timeH, hsH, dirH_data, t02H, tm0m1H
#     # 初始化三维数组
#     data = np.zeros((len(dataL),len(dataL[0][0]) , len(dataL[0][0][0])))
#     dir_data = np.zeros((len(dataL),len(dataL[0][1]) , len(dataL[0][1][0])))
#     t02_data = np.zeros((len(dataL), len(dataL[0][2]), len(dataL[0][2][0])))
#     tm0m1_data = np.zeros((len(dataL),len(dataL[0][3]) , len(dataL[0][3][0])))
#     U10_data = np.zeros((len(dataL), len(dataL[0][4]), len(dataL[0][4][0])))
#     V10_data = np.zeros(( len(dataL),len(dataL[0][5]), len(dataL[0][5][0])))
#     Hi_data= np.zeros(( len(dataH),len(dataH[0][0]), len(dataH[0][0][0])))
#     Hi_dir_data = np.zeros((len(dataH), len(dataH[0][1]), len(dataH[0][1][0])))
#     Hi_t02_data = np.zeros((len(dataH), len(dataH[0][2]), len(dataH[0][2][0])))
#     Hi_tm0m1_data = np.zeros((len(dataH), len(dataH[0][3]), len(dataH[0][3][0])))
#     # Hi_U10_data = np.zeros((len(dataH), len(dataH[0][4]), len(dataH[0][4][0])))
#     # Hi_V10_data = np.zeros((len(dataH), len(dataH[0][5]), len(dataH[0][5][0])))
#
#     # 将数据叠加到三维数组中
#     for i in range(len(dataL)):
#         data[i, :, :] = dataL[i][0]
#         dir_data[i, :, :] = dataL[i][1]
#         t02_data[i, :, :] = dataL[i][2]
#         tm0m1_data[i, :, :] = dataL[i][3]
#         U10_data[i, :, :] = dataL[i][4]
#         V10_data[i, :, :] = dataL[i][5]
#         Hi_data[i, :, :] = dataH[i][0]
#         Hi_dir_data[i, :, :] = dataH[i][1]
#         Hi_t02_data[i, :, :] = dataH[i][2]
#         Hi_tm0m1_data[i, :, :] = dataH[i][3]
#         # Hi_U10_data[i, :, :] = dataH[i][4]
#         # Hi_V10_data[i, :, :] = dataH[i][5]
#     # 只有训练数据需要归一化
#     # data=Zscore(data)
#     # dir_data=Zscore(dir_data)
#     # t02_data=Zscore(t02_data)
#     # tm0m1_data=Zscore(tm0m1_data)
#     # U10_data=Zscore(U10_data)
#     # V10_data=Zscore(V10_data)
# #Hi_U10_data,Hi_V10_data
#
#     return data, dir_data, t02_data, tm0m1_data, U10_data, V10_data, Hi_data,Hi_dir_data, Hi_t02_data,Hi_tm0m1_data,
# def split_and_save_data(data, split_ratio=[0.2, 0.1, 0.7], prefix=''):
#     total_len = len(data)
#     split_points = [int(total_len * ratio) for ratio in split_ratio]
#
#     split_data = []
#     start_idx = 0
#     for split_point in split_points:
#         split_data.append(data[start_idx:start_idx + split_point])
#         start_idx += split_point
#
#     file_names = [prefix + '_train.npy', prefix + '_valid.npy', prefix + '_test.npy']
#     for i in range(len(split_data)):
#         np.save(file_names[i], split_data[i])
#
#     return split_data, file_names
# low_res_path = 'D:\\pythonProject_DT\\WW3lowTrain'
# hi_res_path = 'D:\\pythonProject_DT\\WW3hiTrain'
# # lon_min = 145.15
# # lon_max = 147.65
# # lat_min = -10.05
# # lat_max = -7.55
# lon_min = 135.2 # 最小经度
# lon_max = 170.1  # 最大经度
# lat_min = -44.85  # 最小纬度
# lat_max = 0.3   # 最大纬度
# data, dir_data, t02_data, tm0m1_data, U10_data, V10_data, Hi_data,Hi_dir_data, Hi_t02_data,Hi_tm0m1_data,  = process_data(low_res_path, hi_res_path, lon_min,lon_max,lat_min,lat_max)
# # 定义划分并保存数据的函数Hi_U10_data,Hi_V10_data
#
# # 定义分割比例
# split_ratio = [0.2, 0.1, 0.7]
# # 分别对每个数组进行划分并保存到对应的文件
# data_split, hs_file_names = split_and_save_data(data, split_ratio=split_ratio, prefix='data')
# dir_data_split, dir_file_names = split_and_save_data(dir_data, split_ratio=split_ratio, prefix='dir_data')
# t02_data_split, t02_file_names = split_and_save_data(t02_data, split_ratio=split_ratio, prefix='t02_data')
# tm0m1_data_split, tm0m1_file_names = split_and_save_data(tm0m1_data, split_ratio=split_ratio, prefix= 'tm0m1_data')
# U10_data_split, U10_file_names = split_and_save_data(U10_data, split_ratio=split_ratio, prefix='U10_data')
# V10_data_split, V10_file_names = split_and_save_data(V10_data, split_ratio=split_ratio, prefix='V10_data')
#
# Hi_data_split, Hi_hs_file_names = split_and_save_data(Hi_data, split_ratio=split_ratio, prefix='Hi_data')
# Hi_dir_data_split, Hi_dir_file_names = split_and_save_data(Hi_dir_data, split_ratio=split_ratio, prefix='Hi_dir_data')
# Hi_t02_data_split, Hi_t02_file_names = split_and_save_data(Hi_t02_data, split_ratio=split_ratio, prefix='Hi_t02_data')
# Hi_tm0m1_data_split, Hi_tm0m1_file_names = split_and_save_data(Hi_tm0m1_data, split_ratio=split_ratio, prefix= 'Hi_tm0m1_data')
# # Hi_U10_data_split, Hi_U10_file_names = split_and_save_data(Hi_U10_data, split_ratio=split_ratio, prefix='Hi_U10_data')
# # Hi_V10_data_split, Hi_V10_file_names = split_and_save_data(Hi_V10_data, split_ratio=split_ratio, prefix='Hi_V10_data')
#
# # 输出每个数组的划分结果，供检查
# for file_names, data_split in zip([hs_file_names, dir_file_names, t02_file_names, tm0m1_file_names, U10_file_names, V10_file_names,Hi_hs_file_names], [data_split, dir_data_split, t02_data_split, tm0m1_data_split, U10_data_split, V10_data_split,Hi_data_split]):
#     for i in range(len(file_names)):
#         print(f'{file_names[i]} 分割结果：{data_split[i]}')
# import xarray as xr
# import numpy as np
# import os
# def Zscore(feature):
#     # 分别对每列数据进行归一化
#     feature_normalized = np.zeros_like(feature)  # 创建一个与原数组大小相同的归一化结果数组
#
#     for i in range(feature.shape[1]):  # 对每个列进行迭代
#         for j in range(feature.shape[2]):  # 对每个位置进行迭代
#             column_data = feature[:, i, j]  # 获取当前列数据
#
#             # 计算均值和标准差
#             mean = np.mean(column_data)
#             std = np.std(column_data)
#             # 使用Z-score归一化
#             feature_normalized[:, i, j] = (column_data - mean) / std
#             feature_normalized = np.nan_to_num(feature_normalized, nan=0)
#     return feature_normalized
# def process_data(low_res_path, hi_res_path, lon_min,lon_max,lat_min,lat_max,variable_name,):
#     filelow_list = os.listdir(low_res_path)
#     filelow_list.sort()
#     filehi_list = os.listdir(hi_res_path)
#     filehi_list.sort()
#     dataL = []
#     dataH = []
#     # 读取.nc文件并保存数据
#     for file_name in filelow_list:
#         if file_name.endswith('.nc4'):
#             file_path = os.path.join(low_res_path, file_name)
#             # 裁剪数据
#             dsL = xr.open_dataset(file_path)
#             dsL_subset = dsL.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
#             # 提取数据并存入数据列表
#             timeL = dsL_subset['time'].values
#             hsL = dsL_subset[variable_name].values
#             for i in range(0, len(timeL), 6):
#                 dataL.append([hsL[i]])
#             dsL.close()
#
#     data = np.zeros((len(dataL),len(dataL[0][0]) , len(dataL[0][0][0])))
#
#     # 将数据叠加到三维数组中
#     for i in range(len(dataL)):
#         data[i, :, :] = dataL[i][0]
#     # 只有训练数据需要归一化
#     data=Zscore(data)
#     return data,
# def split_and_save_data(data, split_ratio=[0.2, 0.1, 0.7], prefix=''):
#     total_len = len(data)
#     split_points = [int(total_len * ratio) for ratio in split_ratio]
#
#     split_data = []
#     start_idx = 0
#     for split_point in split_points:
#         split_data.append(data[start_idx:start_idx + split_point])
#         start_idx += split_point
#
#     file_names = [prefix + '_train.npy', prefix + '_valid.npy', prefix + '_test.npy']
#     for i in range(len(split_data)):
#         np.save(file_names[i], split_data[i])
#
#     return split_data, file_names
# low_res_path = 'D:\\pythonProject_DT\\WW3lowTrain'
# hi_res_path = 'D:\\pythonProject_DT\\WW3hiTrain'
#
# lon_min = 135.2 # 最小经度
# lon_max = 152.45  # 最大经度
# lat_min = -11.15  # 最小纬度
# lat_max = 0.3   # 最大纬度
# variable_name=['hs','dir','t02','tm0m1','U10','V10']
# for var in variable_name:
#     data,   = process_data(low_res_path, hi_res_path, lon_min,lon_max,lat_min,lat_max,var)
#     # 定义划分并保存数据的函数Hi_U10_data,Hi_V10_data Hi_data
#     # 定义分割比例
#     split_ratio = [0.2, 0.1, 0.7]
#     # 分别对每个数组进行划分并保存到对应的文件
#     data_split, hs_file_names = split_and_save_data(data, split_ratio=split_ratio, prefix=var+'_data')
import xarray as xr
import numpy as np
import os


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
low_res_path = 'D:\\pythonProject_DT\\WW3lowTrain'
hi_res_path = 'D:\\pythonProject_DT\\WW3hiTrain'
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

