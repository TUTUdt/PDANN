import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.neural_net(x)
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AngleRegressionLoss(nn.Module):#角度损失函数
    def __init__(self):
        super(AngleRegressionLoss, self).__init__()

    def forward(self, out, label):
        loss = 1 - torch.mean(torch.cos((out - label) * np.pi / 180))
        return loss

def reshape_features(features):
    original_shape = features.shape
    new_shape_second_dim = original_shape[1] * original_shape[2]
    features = features.reshape((original_shape[0], new_shape_second_dim))
    return features
def load_data(file_path):
    return np.load(file_path)
def preprocess_data(features, labels):
    nan_indices = np.isnan(labels)
    labels[nan_indices] = 0
    features[np.where((features < -2000) | np.isnan(features))] = 0
    features = reshape_features(features)
    return features, labels