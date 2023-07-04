import os
import random
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, KFold
from torch.nn.utils.rnn import pad_sequence

N_CLIENT = 10

# HAR
# pamap2: 9 workers
# opportunity: 4 workers
# UCI HAR: 30 workers


def collate_fn(batch):
    #batch_data, batch_labels, batch_weight = zip(*batch)
    batch_data, batch_labels = zip(*batch)

    batch_data = pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    #batch_weight = torch.FloatTensor(batch_weight)

    return batch_data, batch_labels#, batch_weight


def load_pamap2(data_path, label_path, client_id, seed=4321):
    target_names = [
        'lying',
        'sitting',
        'standing',
        'walking',
        'running',
        'cycling',
        'nordic-walking',
        'ascending-stairs',
        'descending-stairs',
        'vacuum-cleaning',
        'ironing',
        'rope-jumping'
    ]

    X = pickle.load(open(data_path, 'rb'))
    Y = pickle.load(open(label_path, 'rb'))
    Y = [np.eye(12)[y] for y in Y]

    label_count = np.sum(np.concatenate(Y, axis=0), axis=0)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('total imbalance', imbalance_factor)

    X_train, X_test, Y_train, Y_test, = train_test_split(X[client_id], Y[client_id], test_size=0.2, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed)

    trainData = MyDataset(X_train, Y_train, target_names=target_names)
    valData = MyDataset(X_val, Y_val, target_names=target_names)
    testData = MyDataset(X_test, Y_test, target_names=target_names)

    label_count = np.sum(Y_train, axis=0)
    print('train label count', label_count)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('train imbalance', imbalance_factor)

    label_count = np.sum(Y_test, axis=0)
    print('test label count', label_count)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('test imbalance', imbalance_factor)

    return trainData, valData, testData

class MyDataset(Dataset):
    def __init__(self, X, Y, target_names=None, sample_k=None):
        self.data = np.array(X)
        self.labels = np.array(Y)
        self.target_names = target_names

        self.get_instance_weights()
        self.data_len = np.array([len(x) for x in self.data])

        if sample_k:
            self.random_sample(sample_k)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        #instance_weights = self.instance_weights[idx]
        return data, labels #, instance_weights

    def get_class_weights(self):
        n_classes = len(self.labels[0])
        # Count each class frequency (pos/neg) for each label
        cls_count = np.ones(n_classes)  # avoid nan
        for example_y in self.labels:
            cls_count[example_y.argmax()] += 1

        cls_weight = np.zeros(n_classes)
        for cls in range(n_classes):
            cls_weight[cls] = sum(cls_count) / (cls_count[cls] * len(self.target_names))
        cls_weight[np.isnan(cls_weight)] = 0

        self.instance_weights = []
        for y in self.labels:
            self.instance_weights.append(sum(cls_weight * y))

    def get_instance_weights(self):
        n_classes = len(self.labels[0])
        # Count each class frequency (pos/neg) for each label
        pos_count = np.ones((n_classes))  # avoid nan
        neg_count = np.ones((n_classes))
        sample_size = np.zeros((n_classes))
        for example_y in self.labels:
            for i, y in enumerate(example_y):
                if y == 1:
                    sample_size[i] += 1
                    pos_count[i] += 1
                else:
                    neg_count[i] += 1
        self.num_samples = pos_count - 1
        self.pos_weight = neg_count / (pos_count + neg_count)
        self.neg_weight = pos_count / (pos_count + neg_count)

        self.instance_weights = []
        for y in self.labels:
            weight = (y * self.pos_weight + (1 - y) * self.neg_weight)
            self.instance_weights.append(weight)

        self.instance_weights = np.array(self.instance_weights)

    def random_sample(self, k):
        sample_permuted = np.random.permutation(range(len(self.labels)))[:k]
        self.data = np.array(self.data)[sample_permuted]
        self.data_len = np.array(self.data_len)[sample_permuted]
        self.labels = np.array(self.labels)[sample_permuted]
        self.instance_weights = np.array(self.instance_weights)[sample_permuted]