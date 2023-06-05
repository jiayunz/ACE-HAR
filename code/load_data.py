import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    batch_data, batch_labels = zip(*batch)

    batch_data = pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)

    return batch_data, batch_labels


def load_data(data_path, label_path, seed=4321):
    X = pickle.load(open(data_path, 'rb')) # the data is split by subject
    Y = pickle.load(open(label_path, 'rb'))
    X = np.concatenate(X, axis=0) # concat data from all subjects
    Y = np.concatenate([np.eye(12)[y] for y in Y], axis=0)

    print('loaded X and Y')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed)

    trainData = MyDataset(np.array(X_train), np.array(Y_train))
    valData = MyDataset(np.array(X_val), np.array(Y_val))
    testData = MyDataset(np.array(X_test), np.array(Y_test))

    return trainData, valData, testData


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.data = np.array(X)
        self.labels = np.array(Y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels