import torch 
import os
import csv
import random
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from util import *
from Model.model import *
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, SubsetRandomSampler, Dataset
import torch.nn.functional as F
import torch.nn as nn

matplotlib.use('Agg')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = '../Example_data/NCvsAD'
AV45_PT = os.path.join(DATA_ROOT, 'AV45.pt')
FDG_PT = os.path.join(DATA_ROOT, 'FDG.pt')
VBM_PT = os.path.join(DATA_ROOT, 'VBM.pt')
TRAIN_TEST_PT = os.path.join(DATA_ROOT, 'train_test.pt')

def load_npz_pt(pt_path):
    archive = np.load(pt_path, allow_pickle=True)
    return {key: archive[key] for key in archive.files}

class PackedSplitDataset(Dataset):
    def __init__(self, pt_file, split, adj_dict1, adj_dict2, adj_dict3):
        payload = load_npz_pt(pt_file)
        self.ids = [str(item) for item in payload[f'{split}_ids']]
        self.features = np.asarray(payload[f'{split}_features'])
        self.adj_dict1 = adj_dict1
        self.adj_dict2 = adj_dict2
        self.adj_dict3 = adj_dict3

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample = self.features[idx]
        features = torch.FloatTensor(sample[:-1].astype(float))
        label = int(sample[-1])
        adj_matrix1 = self.adj_dict1[sample_id]
        adj_matrix2 = self.adj_dict2[sample_id]
        adj_matrix3 = self.adj_dict3[sample_id]
        return sample_id, features, label, adj_matrix1, adj_matrix2, adj_matrix3


def pad_or_trim_features(batch_x, target_dim=348):
    if batch_x.size(1) < target_dim:
        padding = torch.zeros(batch_x.size(0), target_dim - batch_x.size(1), device=batch_x.device, dtype=batch_x.dtype)
        return torch.cat([batch_x, padding], dim=1)
    if batch_x.size(1) > target_dim:
        return batch_x[:, :target_dim]
    return batch_x


if True:
    adj1_dict = load_npz_pt(AV45_PT)
    adj2_dict = load_npz_pt(FDG_PT)
    adj3_dict = load_npz_pt(VBM_PT)

    hyperedges, num_nodes = load_hypergraph("../Transcriptome-driven_hypergraph_construction/hypergraph_AD_wgcna/hypergraph_matrix.csv")

    tr_path = TRAIN_TEST_PT
    te_path = TRAIN_TEST_PT
    tr_payload = load_npz_pt(tr_path)
    labels = np.asarray(tr_payload['train_features'])[:, -1].astype(int)
    class_sample_counts = np.bincount(labels)
    weights = 1. / class_sample_counts
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    tr_data = PackedSplitDataset(tr_path, 'train', adj1_dict, adj2_dict, adj3_dict)
    te_data = PackedSplitDataset(te_path, 'test', adj1_dict, adj2_dict, adj3_dict)
    te_data_loader = DataLoader(te_data, batch_size=args.batch_size_, shuffle=False)

    network = Fusion(num_class=2, num_views=3, hidden_dim=[64], dropout=0.2, in_dim=[116, 116, 116], hyperedges=hyperedges)
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)
    class_weights = torch.tensor([class_sample_counts[1] / sum(class_sample_counts), class_sample_counts[0] / sum(class_sample_counts)]).float().to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = best_te_f1 = best_te_auc = best_te_sen = best_te_spe = 0.0

    for epoch in range(2000):

        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        tr_probs_all = []
        tr_labels_all = []
        tr_preds_all = []

        train_loader = DataLoader(
            tr_data,
            batch_size=args.batch_size_,
            sampler=sampler,
            num_workers=4
        )
        network.train()

        for data in train_loader:
            sample_id, batch_x, targets, adj_matrix1, adj_matrix2, adj_matrix3 = data
            B = batch_x.size(0)
            batch_x = pad_or_trim_features(batch_x, target_dim=348)

            def to_tensor(x):
                return x.reshape(-1, 116, 1).float().to(device)

            batch_x1 = to_tensor(batch_x[:, 0:116])
            batch_x2 = to_tensor(batch_x[:, 116:232])
            batch_x3 = to_tensor(batch_x[:, 232:])
            adj1, adj2, adj3 = adj_matrix1.to(device), adj_matrix2.to(device), adj_matrix3.to(device)
            targets = targets.long().to(device)

            optimizer.zero_grad()
            loss_fusion, tr_logits, tr_tcp, *_ = network(batch_x1, batch_x2, batch_x3, adj1, adj2, adj3, targets)
            loss = loss_fusion

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            with torch.no_grad():
                tr_prob = F.softmax(tr_logits, dim=1)
                tr_pre_lab = torch.argmax(tr_prob, 1)
            train_corrects += torch.sum(tr_pre_lab == targets.data)
            train_num += B

            tr_probs_all.extend(tr_prob[:, 1].cpu().numpy())
            tr_labels_all.extend(targets.cpu().numpy())
            tr_preds_all.extend(tr_pre_lab.cpu().numpy())

    # CSV logging block (unchanged
        network.eval()
        test_corrects = test_num = 0
        te_probs_all, te_labels_all, te_preds_all = [], [], []
        with torch.no_grad():
            for data in te_data_loader:
                sample_id, batch_x, targets, adj1, adj2, adj3 = data
                batch_x = pad_or_trim_features(batch_x, target_dim=348)
                batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).float().to(device)
                batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).float().to(device)
                batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).float().to(device)
                targets = targets.long().to(device)
                adj1, adj2, adj3 = adj1.to(device), adj2.to(device), adj3.to(device)
                te_logits = network.infer(batch_x1, batch_x2, batch_x3, adj1, adj2, adj3)

                if torch.isnan(te_logits).any() or torch.isinf(te_logits).any():

                    continue

                te_prob = F.softmax(te_logits, dim=1)
                te_pre_lab = torch.argmax(te_prob, 1)

                test_corrects += torch.sum(te_pre_lab == targets.data)
                test_num += batch_x1.size(0)

                te_probs_all.extend(te_prob[:, 1].cpu().numpy())
                te_labels_all.extend(targets.cpu().numpy())
                te_preds_all.extend(te_pre_lab.cpu().numpy())

        te_probs_all_np = np.array(te_probs_all)
        te_labels_all_np = np.array(te_labels_all)
        valid_mask = ~np.isnan(te_probs_all_np)
        te_probs_all_np = te_probs_all_np[valid_mask]
        te_labels_all_np = te_labels_all_np[valid_mask]

        if len(te_labels_all_np) > 0:
            acc = test_corrects.double().item() / test_num
            auc = roc_auc_score(te_labels_all_np, te_probs_all_np)
            f1 = f1_score(te_labels_all_np, te_preds_all)
            sen = sensitivity_score(te_labels_all_np, te_preds_all)
            spe = specificity_score(te_labels_all_np, te_preds_all)
            print(
                f"Epoch {epoch + 1:04d} | "
                f"train-acc={acc:.4f} train-auc={auc:.4f} train-f1={f1:.4f} train-sen={sen:.4f} train-spe={spe:.4f} | "
                f"test_acc={acc:.4f} test_auc={auc:.4f} test_f1={f1:.4f} "
                f"test_sen={sen:.4f} test_spe={spe:.4f}",
                flush=True,
            )
        else:
            print(f"Epoch {epoch + 1:04d} | no valid test predictions", flush=True)

