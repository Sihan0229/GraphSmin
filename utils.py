import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import pdist, squareform
import tkinter as tk
from sklearn.metrics import accuracy_score

from args import get_parser
args = get_parser().parse_args()


def LT_split(labels, IR, K):
    num_majority = args.LT_major
    C = len(set(labels.tolist()))
    num_sampling = [0] * C
    sorted_sampling = [0] * C
    num_classes = [sum(map(lambda x: x == i, labels.cpu().numpy())) for i in range(C)]
    sorted_indices = sorted(range(len(num_classes)), key=lambda x: num_classes[x], reverse=True)
    de_factor = int(IR) ** (1 / (C - 1))
    sorted_sampling[0] = num_majority
    for i in range(1, C):
        sorted_sampling[i] = sorted_sampling[i - 1] / de_factor
    sorted_sampling = np.around(sorted_sampling).astype(int)
    for i in range(7):
        num_sampling[sorted_indices[i]] = sorted_sampling[i]
    sorted_indices = list(reversed(sorted_indices))[: K]
    return num_sampling, sorted_indices


def step_split(labels, IM, K):
    num_majority = args.ST_major
    C = len(set(labels.tolist()))
    num_minority = int(num_majority / IM)
    num_sampling = [num_majority if i < C - K else num_minority for i in range(C)]
    return num_sampling


def split_data(labels, IR, K):
    num_classes = len(set(labels.tolist()))
    c_idxs = []
    train_idx = []
    test_idx = []
    sorted_indices = []

    if args.split_method == 'LT_sampling':
        c_train_num, sorted_indices = LT_split(labels, IR, K)
    else:
        c_train_num = step_split(labels, IR, K)

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:, -1].tolist()
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        test_idx = test_idx + c_idx[c_train_num[i]:c_train_num[i] + 30]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)
    # print("c_train_num: ", c_train_num)
    # print("sorted_indices: ", sorted_indices)
    # print()
    if args.split_method == 'LT_sampling':
        return train_idx, test_idx, sorted_indices
    else:
        return train_idx, test_idx


def split_data_normal(labels):
    num_classes = len(set(labels.tolist()))

    trains = []
    tests = []

    for i in range(num_classes):
        class_items = (labels == i).nonzero().squeeze().tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(class_items)))
        random.shuffle(class_items)
        total_length = len(class_items)

        # 7:3
        split_1 = int(total_length * 0.7)

        list_1 = class_items[:split_1]
        list_2 = class_items[split_1:]

        trains.append(list_1)
        tests.append(list_2)
    random.shuffle(trains)
    trains = list(tk._flatten(trains))
    tests = list(tk._flatten(tests))

    trains = torch.LongTensor(trains).view(-1, 1).squeeze()
    tests = torch.LongTensor(tests).view(-1, 1).squeeze()

    return trains, tests


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def print_class_acc(output, labels):
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.cpu().detach().numpy(), F.softmax(output.cpu().detach(), dim=-1).numpy(),
                                  average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:, 1].detach(), average='macro')

    macro_F = f1_score(labels.cpu().detach().numpy(), torch.argmax(output.cpu().detach(), dim=-1).numpy(),
                       average='macro')
    print(' auc-roc score: {:.4f}, macro_F score: {:.4f}'.format(auc_score, macro_F))
    return auc_score, macro_F


def step_sample(embed, labels, idx_train, adj=None, portion=1.0, num_minority=3):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new = None

    for i in range(num_minority):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def long_tailed_sampling(embed, labels, idx_train, adj=None, portion=0.8, sorted_indices=None):
    im_classes = sorted_indices
    adj_new = None

    for key, value in enumerate(im_classes):
        chosen = idx_train[(labels == value)[idx_train]]
        num = int(chosen.shape[0] * portion)
        chosen = chosen[:num]
        chosen_embed = embed[chosen, :]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance, distance.max() + 100)
        idx_neighbor = distance.argmin(axis=-1)
        interp_place = random.random()

        new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
        new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(value)
        idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
        idx_train_append = idx_train.new(idx_new)

        embed = torch.cat((embed, new_embed), 0)
        labels = torch.cat((labels, new_labels), 0)
        idx_train = torch.cat((idx_train, idx_train_append), 0)

        if adj is not None:
            if adj_new is None:
                adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
            else:
                temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:, :]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:, :]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:, :]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def adj_mse_loss(adj_rec, adj_tgt):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0] ** 2
    neg_weight = edge_num / (total_num - edge_num)
    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt == 0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)
    return loss


def augmentation(features_1, adj_1, features_2, adj_2, training):
    # view 1
    mask_1, _ = get_feat_mask(features_1, args.maskfeat_rate_1)
    features_1 = features_1 * (1 - mask_1)
    adj_1 = F.dropout(adj_1, p=args.dropedge_rate_1, training=training)

    # view 2
    mask_2, _ = get_feat_mask(features_1, args.maskfeat_rate_2)
    features_2 = features_2 * (1 - mask_2)
    adj_2 = F.dropout(adj_2, p=args.dropedge_rate_2, training=training)

    return features_1, adj_1, features_2, adj_2


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask, samples


def normalize_lp_adj(adj):
    adj += torch.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()
    adj = torch.from_numpy(adj)

    return adj


def normalize_hp_adj(adj):
    adj += torch.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()
    adj = torch.from_numpy(adj)
    adj = torch.eye(adj.shape[0]) - adj * 0.1  # I - (D-1/2)A˜(D-1/2) * a
    return adj


def homo_score(labels, adj):
    score_hm = 0
    score_ht = 0
    for i in range(1, adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 0:
                continue
            if labels[i] == labels[j]:
                score_hm += 1
            else:
                score_ht += 1
    count_ones = np.count_nonzero(adj.cpu().numpy() == 1) / 2
    return score_hm / count_ones









