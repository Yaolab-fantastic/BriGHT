import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
import os
import logging
import time
import csv
import codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import sklearn.metrics
import torch_geometric
from scipy.sparse import coo_matrix
import torch.nn.functional as F

from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import csv, torchvision, numpy as np, random, os
from PIL import Image

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]

            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                #pair_indices.append(random.choice(self.dataset.classwise_indices[y]))
                if hasattr(self.dataset, 'classwise_indices_tcp') and \
                    y in self.dataset.classwise_indices_tcp and \
                    len(self.dataset.classwise_indices_tcp[y]) > 0:
                        pair_indices.append(random.choice(self.dataset.classwise_indices_tcp[y]))
                else:
                        pair_indices.append(random.choice(self.dataset.classwise_indices[y]))


            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels
        
        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)

        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


class CustomDatasetWithAdj(Dataset):
    def __init__(self, csv_file, adj_dict1, adj_dict2, adj_dict3):
        self.data = pd.read_csv(csv_file).values
        self.adj_dict1 = adj_dict1
        self.adj_dict2 = adj_dict2
        self.adj_dict3 = adj_dict3


        self.targets = self.data[:, -1].astype(int).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_id = sample[0]
        features = torch.FloatTensor(sample[1:-1].astype(float))  # [348]
        label = int(sample[-1]) 
        adj_matrix1 = self.adj_dict1[sample_id]
        adj_matrix2 = self.adj_dict2[sample_id]
        adj_matrix3 = self.adj_dict3[sample_id]
        return sample_id, features, label, adj_matrix1, adj_matrix2, adj_matrix3


def load_adj_data(folder_path,x):
    file_names = os.listdir(folder_path)
    adj_dict = {}
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        adj_data = pd.read_csv(file_path, header=None).values.astype(float)
        adj_data = torch.LongTensor(np.where(adj_data > x, 1, 0))
        file_name_without_extension = os.path.splitext(file_name)[0]
        adj_dict[file_name_without_extension] = adj_data
    return adj_dict


def specificity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spe
def sensitivity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sen

################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

################
# subgraph
################
def adj_to_PyG_edge_index(adj):
    coo_A = coo_matrix(adj)
    edge_index, edge_weight = torch_geometric.utils.convert.from_scipy_sparse_matrix(coo_A)
    return edge_index

def data_to_PyG_data(x, edge_index, y):
    out_data = x
    out_edge_index = edge_index
    out_label = y
    PyG_data = torch_geometric.data.Data(x=out_data, edge_index=out_edge_index, y=out_label)
    return PyG_data

def PyG_edge_index_to_adj(edge_index):
    adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index)
    return adj

def data_write_csv(file_name, datas):
  file_csv = codecs.open(file_name,'w+','utf-8')
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("doc saved")

################
# hypergraph_laplacian
################
def hypergraph_laplacian(hyperedges, num_nodes):

    num_edges = len(hyperedges)
    H = torch.zeros((num_nodes, num_edges), device=device)  
    
    for e, edge in enumerate(hyperedges):
        for v in edge:
            H[v, e] = 1  
    
    node_degrees = H.sum(dim=1)  
    edge_degrees = H.sum(dim=0) 
    
    eps = 1e-8
    node_degrees_safe = node_degrees + eps  
    edge_degrees_safe = edge_degrees + eps  
    

    Dv_sqrt_inv = torch.diag(1.0 / torch.sqrt(node_degrees_safe))  
    De_inv = torch.diag(1.0 / edge_degrees_safe)  
    

    L = torch.eye(num_nodes, device=device) - Dv_sqrt_inv @ H @ De_inv @ H.T @ Dv_sqrt_inv
    
    return L


def load_hypergraph(file_path):

    df = pd.read_csv(file_path)
    num_nodes = len(df)
    hyperedges = []
    for col in df.columns[1:]:
        edge_nodes = df.index[df[col] == 1].tolist()
        hyperedges.append(edge_nodes)
    
    return hyperedges, num_nodes