from sympy import print_glsl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
from preprocess.preprocessembeddings import name_mapping_text, name_mapping_audio
import json
import random
from collections import defaultdict
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'ADReSSo21', 'diagnosis', 'train')
splits_dir = os.path.join(DATA_ROOT, 'splits')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length_wav2vec = 4000


class AdressoDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def smart_pt_load(base_path, must_contain=None):
    if os.path.exists(base_path):
        return torch.load(base_path).to(device)
    dirname = os.path.dirname(base_path)
    fname_prefix = os.path.basename(base_path).split('.')[0][:4]  # 比如09-1
    candidates = [f for f in os.listdir(dirname) if
        fname_prefix in f and f.endswith('.pt') and (must_contain is None or must_contain in f)]
    if candidates:
        print(f"[WARN] {base_path} not found! Using {candidates[0]}")
        return torch.load(os.path.join(dirname, candidates[0])).to(device)

    raise FileNotFoundError(f"Cannot find any pt file similar to {base_path}")


def read_CSV(config, augment=False):
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'ADReSSo21', 'diagnosis', 'train')
    root_text_path = os.path.join(DATA_ROOT, 'text')
    csv_labels_path = os.path.join(DATA_ROOT, 'adresso-train-mmse-scores.csv')
    if augment:
        DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'ADReSSo21', 'diagnosis', 'train_aug')
        root_text_path = os.path.join(DATA_ROOT, 'text')
        csv_labels_path = os.path.join(DATA_ROOT, 'adresso-train-mmse-scores_aug.csv')
    labels_pd = pd.read_csv(csv_labels_path)

    uids = []
    features = []
    labels = []

    pauses_data = '_pauses' if config.model.pauses else ''
    audio_model_name = config.model.audio_model
    text_model_name = config.model.textual_model

    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        labels.append(torch.tensor(0 if row['dx'] == "cn" else 1).to(device).float())

        if text_model_name != '':
            text_file_name = row['adressfname'] + '_' + name_mapping_text[text_model_name] + pauses_data + '.pt'
            text_embeddings_path = os.path.join(root_text_path, row['dx'], text_file_name)

        if audio_model_name != '':
            audio_file_name = row['adressfname'] + '_' + name_mapping_text[text_model_name] + pauses_data + '_' + \
                              name_mapping_audio[audio_model_name] + '.pt'
            audio_embeddings_path = os.path.join(root_text_path, row['dx'], audio_file_name)

        if config.model.multimodality:
            features.append((smart_pt_load(audio_embeddings_path, must_contain=f'_audio_{audio_model_name}'),
                             smart_pt_load(text_embeddings_path, must_contain=name_mapping_text[text_model_name])))
        else:
            if text_model_name != '':
                features.append(smart_pt_load(text_embeddings_path, must_contain=name_mapping_text[text_model_name]))
            elif audio_model_name != '':
                features.append(smart_pt_load(audio_embeddings_path, must_contain=f'_audio_{audio_model_name}'))

    return uids, features, labels


def get_cl_dataloaders(config):
    uids, features, labels = read_CSV(config)
    batch_size = config.train.batch_size

    # 集中训练：8:2 随机划分 train/val
    train_feat, val_feat, train_labels, val_labels = train_test_split(features, labels, test_size=0.2,
        random_state=3407, stratify=labels)

    train_dataset = AdressoDataset(train_feat, train_labels)
    val_dataset = AdressoDataset(val_feat, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_local_dataloaders_cv(config, num_clients=3, kfold_number=0):
    # 此处feature与label都是list，长度都为165，也就是数据集样本数量
    augment = config.train.augment if config.train.augment else False
    uids, features, labels = read_CSV(config)
    batch_size = config.train.batch_size

    # ✅ 将 features 和 labels 转换为 NumPy 数组（如果它们是 GPU tensor）
    if isinstance(features[0], torch.Tensor):
        features = [f.cpu().numpy() for f in features]
    if isinstance(labels[0], torch.Tensor):
        labels = [l.cpu().numpy() for l in labels]

    validation_split = np.load(os.path.join(splits_dir, f'val_uids{kfold_number}.npy'))
    train_uids = []
    train_features = []
    train_labels = []

    validation_uids = []
    validation_features = []
    validation_labels = []

    for i in range(len(uids)):
        if uids[i] in validation_split:
            validation_uids.append(uids[i])
            validation_features.append(features[i])
            validation_labels.append(labels[i])
        else:
            train_uids.append(uids[i])
            train_features.append(features[i])
            train_labels.append(labels[i])

    if augment:

        uids_aug, features_aug, labels_aug = read_CSV(config, augment=True)
        if isinstance(features_aug[0], torch.Tensor):
            features_aug = [f.cpu().numpy() for f in features_aug]
        if isinstance(labels_aug[0], torch.Tensor):
            labels_aug = [l.cpu().numpy() for l in labels_aug]

        validation_speaker_ids = set()
        for uid in validation_uids:
            # 解析原始uid，例如 "adrso234" -> speaker_id 234
            if 'adrso' in uid:
                try:
                    speaker_id = int(uid.replace('adrso', ''))
                    validation_speaker_ids.add(speaker_id)
                except ValueError:
                    continue

        validation_aug_uids = set()
        for uid_aug in uids_aug:
            # 解析增广uid，例如 "adrso_234_156" -> speaker_id 234, text_id 156
            parts = uid_aug.split('_')
            if len(parts) >= 3:
                try:
                    speaker_id = int(parts[1])
                    text_id = int(parts[2])
                    if speaker_id in validation_speaker_ids or text_id in validation_speaker_ids:  # 去重speaker id 与 text id
                        validation_aug_uids.add(uid_aug)
                except (ValueError, IndexError):
                    continue

        filtered_aug_features = []
        filtered_aug_labels = []
        filtered_aug_uids = []
        # 过滤增广数据，只保留训练集对应的增广数据
        for i in range(len(uids_aug)):
            aug_uid = uids_aug[i]
            # 如果增广uid不在验证集的增广uid中，则保留该增广数据
            if aug_uid not in validation_aug_uids:
                filtered_aug_features.append(features_aug[i])
                filtered_aug_labels.append(labels_aug[i])
                filtered_aug_uids.append(uids_aug[i])


        train_features.extend(filtered_aug_features)
        train_labels.extend(filtered_aug_labels)
        train_uids.extend(filtered_aug_uids)

    train_features, train_labels, train_uids = shuffle(train_features, train_labels, train_uids, random_state=3407)

    total = len(train_features)
    client_size = total // num_clients
    client_dataloarder_list = []

    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size if i != num_clients - 1 else total
        client_feat = train_features[start:end]
        client_lab = train_labels[start:end]
        client_uids = train_uids[start:end]

        client_dataset = AdressoDataset(client_feat, client_lab)
        client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_dataloarder_list.append(client_dataloader)

    validation_dataset = AdressoDataset(validation_features, validation_labels)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return client_dataloarder_list, validation_dataloader


def get_local_dataloaders(config, num_clients=3, **kwargs):
    uids, features, labels = read_CSV(config)

    batch_size = config.train.batch_size

    if isinstance(features[0], torch.Tensor):
        features = [f.cpu().numpy() for f in features]
    if isinstance(labels[0], torch.Tensor):
        labels = [l.cpu().numpy() for l in labels]

    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2,
        random_state=3407, stratify=labels)

    train_features, train_labels = shuffle(train_features, train_labels, random_state=3407)

    total = len(train_features)
    client_size = total // num_clients

    local_dataloaders = []

    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size if i != num_clients - 1 else total

        client_feat = train_features[start:end]
        client_lab = train_labels[start:end]

        dataset = AdressoDataset(client_feat, client_lab)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        local_dataloaders.append(dataloader)

    val_dataset = AdressoDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return local_dataloaders, val_loader


def get_dataloaders(config, kfold_number=0):
    uids, features, labels = read_CSV(config)

    validation_split = np.load(os.path.join(splits_dir, f'val_uids{kfold_number}.npy'))
    batch_size = config.train.batch_size
    train_uids = []
    train_features = []
    train_labels = []

    validation_uids = []
    validation_features = []
    validation_labels = []

    for i in range(len(uids)):
        if uids[i] in validation_split:
            validation_uids.append(uids[i])
            validation_features.append(features[i])
            validation_labels.append(labels[i])
        else:
            train_uids.append(uids[i])
            train_features.append(features[i])
            train_labels.append(labels[i])

    if config.train.augment:
        uids_aug, features_aug, labels_aug = read_CSV(config, augment=True)
        validation_speaker_ids = set()
        for uid in validation_uids:
            if 'adrso' in uid:
                try:
                    speaker_id = int(uid.replace('adrso', ''))
                    validation_speaker_ids.add(speaker_id)
                except ValueError:
                    continue

        validation_aug_uids = set()
        for uid_aug in uids_aug:
            parts = uid_aug.split('_')
            if len(parts) >= 3:
                try:
                    speaker_id = int(parts[1])
                    text_id = int(parts[2])
                    if speaker_id in validation_speaker_ids or text_id in validation_speaker_ids:  # 去重speaker id 与 text id
                        validation_aug_uids.add(uid_aug)
                except (ValueError, IndexError):
                    continue

        filtered_aug_features = []
        filtered_aug_labels = []
        filtered_aug_uids = []
        for i in range(len(uids_aug)):
            aug_uid = uids_aug[i]
            if aug_uid not in validation_aug_uids:
                filtered_aug_features.append(features_aug[i])
                filtered_aug_labels.append(labels_aug[i])
                filtered_aug_uids.append(uids_aug[i])


        train_features.extend(filtered_aug_features)
        train_labels.extend(filtered_aug_labels)
        train_uids.extend(filtered_aug_uids)


    train_dataset = AdressoDataset(train_features, train_labels)
    validation_dataset = AdressoDataset(validation_features, validation_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader


def set_splits():
    csv_labels_path = os.path.join(DATA_ROOT, 'adresso-train-mmse-scores.csv')
    labels_pd = pd.read_csv(csv_labels_path)
    uids = []
    labels = []
    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        labels.append(0 if row['dx'] == "cn" else 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3407)

    for i, (train_index, test_index) in enumerate(skf.split(uids, labels)):
        print("TRAIN:", train_index, "\nTEST:", test_index, '\n')
        np.save(os.path.join(splits_dir, f'train_uids{i}.npy'), np.array(uids)[train_index])
        np.save(os.path.join(splits_dir, f'val_uids{i}.npy'), np.array(uids)[test_index])  ###


def get_splits_stats():
    csv_labels_path = os.path.join(DATA_ROOT, 'adresso-train-mmse-scores.csv')
    labels_pd = pd.read_csv(csv_labels_path)
    uids = []

    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])

    for i in range(5):
        training_split = np.load(os.path.join(splits_dir, f'train_uids{i}.npy'))
        validation_split = np.load(os.path.join(splits_dir, f'val_uids{i}.npy'))
        n_cn_train = 0
        n_ad_train = 0
        n_cn_val = 0
        n_ad_val = 0

        for uid in training_split:
            if labels_pd[labels_pd['adressfname'] == uid]['dx'].values[0] == 'cn':
                n_cn_train += 1
            else:
                n_ad_train += 1

        for uid in validation_split:
            if labels_pd[labels_pd['adressfname'] == uid]['dx'].values[0] == 'cn':
                n_cn_val += 1
            else:
                n_ad_val += 1

        print(f"Fold {i}:")
        print(f"Training CN: {n_cn_train}, Training AD: {n_ad_train}")
        print(f"Validation CN: {n_cn_val}, Validation AD: {n_ad_val}")


if __name__ == '__main__':
    set_splits()
