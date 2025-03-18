import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from ANN import Net,AngleRegressionLoss, load_data, preprocess_data, reshape_features
from ANN import CustomDataset
import torch.nn as nn
def train_model(model, train_loader, validation_dataset, loss_func, optimizer, num_epochs, tra_mse_bench, no_improve_count_t, val_mse_bench, no_improve_count_v, learning_rate, mse_acc):
    model.train()
    model.cuda(0)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.cuda(0)
            labels = labels.cuda(0)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            mse_acc = np.append(mse_acc, loss.item())

        model.eval()
        with torch.no_grad():
            val_mse = loss_func(model(Variable(validation_dataset.features).cuda(0)), Variable(validation_dataset.labels).cuda(0))

        if val_mse > val_mse_bench:
            no_improve_count_v += 1
        else:
            val_mse_bench = val_mse
            no_improve_count_v = 0
        if no_improve_count_v > 20:
            print('Validation loss has not improved for 10 epochs. Stopping training.')
            break
        if mse_acc.mean() > tra_mse_bench:
            no_improve_count_t += 1
        else:
            tra_mse_bench = mse_acc.mean()
            no_improve_count_t = 0
        if (no_improve_count_t > 3) and (learning_rate > 0.0004):
            learning_rate *= 0.75
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            no_improve_count_t = 0

        if epoch % 10 == 0:
            print('[%d] loss: %.4f   %.4f ' % (epoch + 1, np.sqrt(mse_acc.mean()), torch.sqrt(val_mse)))

def main():
    folder_number = ['1', '2', '3', '4', '5', '6', '7', '8']
    variable_name = ['hs', 'dir', 't02', 'tm0m1']

    for var in variable_name:
        for folder in folder_number:
            hs_data_train = load_data(f'{folder}\\OZCdata\\hsZ_data_train.npy')
            dir_data_train = load_data(f'{folder}\\OZCdata\\dirZ_data_train.npy')
            t02_data_train = load_data(f'{folder}\\OZCdata\\t02Z_data_train.npy')
            tm0m1_data_train = load_data(f'{folder}\\OZCdata\\tm0m1Z_data_train.npy')
            U10_data_train = load_data(f'{folder}\\OZCdata\\U10Z_data_train.npy')
            V10_data_train = load_data(f'{folder}\\OZCdata\\V10Z_data_train.npy')
            Hi_hs_data_train = load_data(f'{folder}\\OWW3Cdata\\Hi_{var}_data_train.npy')

            hs_data_V = load_data(f'{folder}\\OZCdata\\hsZ_data_valid.npy')
            dir_data_V = load_data(f'{folder}\\OZCdata\\dirZ_data_valid.npy')
            t02_data_V = load_data(f'{folder}\\OZCdata\\t02Z_data_valid.npy')
            tm0m1_data_V = load_data(f'{folder}\\OZCdata\\tm0m1Z_data_valid.npy')
            U10_data_V = load_data(f'{folder}\\OZCdata\\U10Z_data_valid.npy')
            V10_data_V = load_data(f'{folder}\\OZCdata\\V10Z_data_valid.npy')
            Hi_hs_data_V = load_data(f'{folder}\\OWW3Cdata\\Hi_{var}_data_valid.npy')

            feature_train = np.zeros((hs_data_train.shape[0], 4, 6))
            feature_V = np.zeros((hs_data_V.shape[0], 4, 6))
            target_train = np.zeros((hs_data_train.shape[0], 1))
            target_V = np.zeros((hs_data_V.shape[0], 1))

            for r in range(hs_data_train.shape[1] - 1):
                for j in range(hs_data_train.shape[2] - 1):
                    for i in range(4):
                        if i < 2:
                            feature_train[:, i, 0] = hs_data_train[:, r, j + i]
                            feature_train[:, i, 1] = dir_data_train[:, r, j + i]
                            feature_train[:, i, 2] = t02_data_train[:, r, j + i]
                            feature_train[:, i, 3] = tm0m1_data_train[:, r, j + i]
                            feature_train[:, i, 4] = U10_data_train[:, r, j + i]
                            feature_train[:, i, 5] = V10_data_train[:, r, j + i]

                            feature_V[:, i, 0] = hs_data_V[:, r, j + i]
                            feature_V[:, i, 1] = dir_data_V[:, r, j + i]
                            feature_V[:, i, 2] = t02_data_V[:, r, j + i]
                            feature_V[:, i, 3] = tm0m1_data_V[:, r, j + i]
                            feature_V[:, i, 4] = U10_data_V[:, r, j + i]
                            feature_V[:, i, 5] = V10_data_V[:, r, j + i]
                        else:
                            feature_train[:, i, 0] = hs_data_train[:, r + 1, j + i - 2]
                            feature_train[:, i, 1] = dir_data_train[:, r + 1, j + i - 2]
                            feature_train[:, i, 2] = t02_data_train[:, r + 1, j + i - 2]
                            feature_train[:, i, 3] = tm0m1_data_train[:, r + 1, j + i - 2]
                            feature_train[:, i, 4] = U10_data_train[:, r + 1, j + i - 2]
                            feature_train[:, i, 5] = V10_data_train[:, r + 1, j + i - 2]

                            feature_V[:, i, 0] = hs_data_V[:, r + 1, j + i - 2]
                            feature_V[:, i, 1] = dir_data_V[:, r + 1, j + i - 2]
                            feature_V[:, i, 2] = t02_data_V[:, r + 1, j + i - 2]
                            feature_V[:, i, 3] = tm0m1_data_V[:, r + 1, j + i - 2]
                            feature_V[:, i, 4] = U10_data_V[:, r + 1, j + i - 2]
                            feature_V[:, i, 5] = V10_data_V[:, r + 1, j + i - 2]

                    for a in range(6 * r, 7 + 6 * r):
                        for b in range(6 * j, 7 + 6 * j):
                            if a < 7 + 6 * r and b < 7 + 6 * j:
                                target_train[:, 0] = Hi_hs_data_train[:, a, b]
                                target_V[:, 0] = Hi_hs_data_V[:, a, b]

                                if np.all((target_train == 0) | np.isnan(target_train)):
                                    continue

                                features, labels = preprocess_data(feature_train, target_train)
                                featuresV, labelsV = preprocess_data(feature_V, FB_V)

                                if features.sum() == 0 or np.isnan(features).all():
                                    features = featureup

                                train_dataset = CustomDataset(features, labels)
                                validation_dataset = CustomDataset(featuresV, labelsV)
                                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

                                input_size = features.shape[1]
                                hidden_size = 16
                                output_size = 1
                                num_epochs = 500
                                learning_rate = 0.004
                                val_mse_bench = 1000000.
                                tra_mse_bench = 1000000.
                                no_improve_count_v = 0
                                no_improve_count_t = 0
                                model = Net(input_size, hidden_size, output_size)
                                if var == 'dir':
                                    loss_func = AngleRegressionLoss()
                                else:
                                    loss_func = nn.MSELoss()
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                mse_acc = np.array([])
                                train_model(model, train_loader, validation_dataset, loss_func, optimizer, num_epochs, tra_mse_bench, no_improve_count_t, val_mse_bench, no_improve_count_v, learning_rate, mse_acc)
                                model = model.to(torch.device("cpu"))
                                torch.save(model.state_dict(), f'{folder}\\NET-{var}\\Net{a}+{b}.pt')
                                featureup = feature_train

if __name__ == "__main__":
    main()
