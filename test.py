import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from ANN import Net, load_data, reshape_features, CustomDataset
import torch.nn as nn
def test(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.tolist())
            true_labels.extend(labels.tolist())

    return np.array(predictions), np.array(true_labels)

def main():
    folder_number = ['1', '2', '3', '4', '5', '6', '7', '8']
    variable_name = ['hs']

    for var in variable_name:
        for folder in folder_number:
            hs_data = load_data(f'{folder}\\OZCdata\\hsZ_data_test.npy')
            dir_data = load_data(f'{folder}\\OZCdata\\dirZ_data_test.npy')
            t02_data = load_data(f'{folder}\\OZCdata\\t02Z_data_test.npy')
            tm0m1_data = load_data(f'{folder}\\OZCdata\\tm0m1Z_data_test.npy')
            U10_data = load_data(f'{folder}\\OZCdata\\U10Z_data_test.npy')
            V10_data = load_data(f'{folder}\\OZCdata\\V10Z_data_test.npy')

            target_data = load_data(f'{folder}\\OWW3Cdata\\Hi_{var}_data_test.npy')

            shape = target_data.shape
            rmse_array = np.zeros(shape[1:])
            bias_array = np.zeros(shape[1:])
            r_array = np.zeros(shape[1:])

            for a in range(shape[1]):
                for b in range(shape[2]):
                    if np.all((target_data[:, a, b] == 0) | np.isnan(target_data[:, a, b])):
                        continue

                    features = np.stack([
                        hs_data[:, a, b],
                        dir_data[:, a, b],
                        t02_data[:, a, b],
                        tm0m1_data[:, a, b],
                        U10_data[:, a, b],
                        V10_data[:, a, b]
                    ], axis=1)
                    features = reshape_features(features)

                    labels = target_data[:, a, b].reshape(-1, 1)
                    features[np.isnan(features)] = 0
                    labels[np.isnan(labels)] = 0

                    test_dataset = CustomDataset(features, labels)
                    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

                    model_path = f'{folder}\\NET-{var}\\Net{a}+{b}.pt'
                    predictions, true_labels = test(Net(features.shape[1], 16, 1), test_loader, model_path)
                    #对角度数据做角度矫正
                    if var == 'dir':
                        predictions = (predictions + 180) % 360
                        predictions[np.where((true_labels - predictions) > 180)[0]] = (
                                predictions[np.where((true_labels - predictions) > 180)[0]] + 360
                        )
                        true_labels[np.where((true_labels - predictions) < -180)[0]] = (
                                true_labels[np.where((true_labels - predictions) < -180)[0]] + 360
                        )

                    bias_array[a, b] = np.mean(predictions - true_labels)
                    rmse_array[a, b] = np.sqrt(mean_squared_error(true_labels, predictions))
                    r_array[a, b], _ = pearsonr(true_labels.flatten(), predictions.flatten())

            np.save(f'{folder}\\Metrics\\{var}test_RMSEdownscale.npy', rmse_array)
            np.save(f'{folder}\\Metrics\\{var}test_Biasdownscale.npy', bias_array)
            np.save(f'{folder}\\Metrics\\{var}test_Rdownscale.npy', r_array)

if __name__ == "__main__":
    main()