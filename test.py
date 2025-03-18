import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import pearsonr
from ANN import Net, reshape_features, CustomDataset
import time

def ANNtest(model, test_loader, a, b, folder, var):
    """
    测试模型并计算相关指标
    :param model: 待测试的模型
    :param test_loader: 测试数据加载器
    :param a: 参数a
    :param b: 参数b
    :param folder: 文件夹路径
    :param var: 变量名
    :return: 预测值和真实标签值
    """
    model.load_state_dict(torch.load(f'{folder}\\NET-{var}\\Net{a}+{b}.pt'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        predictions = []
        true_labels = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Test function execution time: {end_time - start_time} seconds")
            predictions.extend(outputs.tolist())
            true_labels.extend(labels.tolist())

    mse = np.sqrt(mean_squared_error(true_labels, predictions))
    predictionarry = [value for row in predictions for value in row]
    true_labelsarry = [value for row in true_labels for value in row]
    return predictionarry, true_labelsarry



folder_numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
variable_names = ['hs', 't02', 'tm0m1', 'dir']

for var in variable_names:
    for folder in folder_numbers:
        data_files = [
            f'{folder}\\OZCdata\\hsZ_data_test.npy',
            f'{folder}\\OZCdata\\dirZ_data_test.npy',
            f'{folder}\\OZCdata\\t02Z_data_test.npy',
            f'{folder}\\OZCdata\\tm0m1Z_data_test.npy',
            f'{folder}\\OZCdata\\U10Z_data_test.npy',
            f'{folder}\\OZCdata\\V10Z_data_test.npy',
            f'{folder}\\OWW3Cdata\\Hi_{var}_data_test.npy'
        ]
        hs_data_T, dir_data_T, t02_data_T, tm0m1_data_T, U10_data_T, V10_data_T, Hi_hs_data_T = [
            np.array(np.load(file)) for file in data_files
        ]

        length_dim1, length_dim2, length_dim3 = hs_data_T.shape
        feature_T = np.zeros((length_dim1, 4, 6))#4 代表就近的四个点位，6代表6种特征量
        featureup = np.zeros((length_dim1, 4, 6))#用来记录某一高分辨率点位对应的上一个低分辨率单位的特征数据
        target_T = np.zeros((length_dim1, 1))

        shapehi = Hi_hs_data_T.shape
        length_hi_dim2, length_hi_dim3 = shapehi[1:]
        RMSEARRY = np.zeros((length_hi_dim2, length_hi_dim3))
        BiasARRY = np.zeros((length_hi_dim2, length_hi_dim3))
        RARRY = np.zeros((length_hi_dim2, length_hi_dim3))

        for r in range(length_dim2 - 1):
            for j in range(length_dim3 - 1):
                for i in range(4):
                    row = r if i < 2 else r + 1
                    col = j + i if i < 2 else j + i - 2
                    feature_T[:, i, 0] = hs_data_T[:, row, col]
                    feature_T[:, i, 1] = dir_data_T[:, row, col]
                    feature_T[:, i, 2] = t02_data_T[:, row, col]
                    feature_T[:, i, 3] = tm0m1_data_T[:, row, col]
                    feature_T[:, i, 4] = U10_data_T[:, row, col]
                    feature_T[:, i, 5] = V10_data_T[:, row, col]

                for a in range(6 * r, 7 + 6 * r):
                    for b in range(6 * j, 7 + 6 * j):
                        if a < 7 + 6 * r and b < 7 + 6 * j:
                            target_T[:, 0] = Hi_hs_data_T[:, a, b]
                            if np.all((target_T == 0) | np.isnan(target_T)):
                                continue

                            labels = target_T
                            labels = np.nan_to_num(labels, nan=0)
                            features = feature_T
                            features = np.where((features < -2000) | np.isnan(features), 0, features)
                            features = reshape_features(features)

                            if features.sum() == 0 or np.isnan(features).all():
                                features = featureup
                            else:
                                test_dataset = CustomDataset(features, labels)
                                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                                input_size = features.shape[1]
                                hidden_size = 16
                                output_size = 1
                                model = Net(input_size, hidden_size, output_size)
                                predictionarry, true_labelsarry = ANNtest(model, test_loader, a, b, folder, var)

                            predictionarry = np.array(predictionarry)
                            true_labelsarry = np.array(true_labelsarry)

                            if var == 'dir':
                                predictionarry = (predictionarry + 180) % 360
                                large_diff_indices = (true_labelsarry - predictionarry) > 180
                                small_diff_indices = (true_labelsarry - predictionarry) < -180
                                predictionarry[large_diff_indices] += 360
                                true_labelsarry[small_diff_indices] += 360

                            bias = np.mean(predictionarry - true_labelsarry)
                            BiasARRY[a, b] = bias
                            rmse = np.sqrt(mean_squared_error(true_labelsarry, predictionarry))
                            RMSEARRY[a, b] = rmse
                            cc, _ = pearsonr(true_labelsarry, predictionarry)
                            RARRY[a, b] = cc

        np.save(f'{folder}\\Metrics\\{var}test_RMSEdownscale.npy', RMSEARRY)
        np.save(f'{folder}\\Metrics\\{var}test_Biasdownscale.npy', BiasARRY)
        np.save(f'{folder}\\Metrics\\{var}test_Rdownscale.npy', RARRY)
