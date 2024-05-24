import numpy as np
import torch
from sklearn.metrics import pairwise_distances as pair
import torch.nn.functional as F
import utils
from args import get_parser

args = get_parser().parse_args()
top_k = args.topK


def construct_graph(features, labels):
    print("constructing KNN graph...")
    [nnodes, n_feat] = features.shape
    adj_hm, adj_ht = con_graph(features, labels, nnodes)

    adj_hm = build_adjacency_matrix(adj_hm, False)
    adj_ht = build_adjacency_matrix(adj_ht, False)

    adj_hm = adj_hm.to(features.device)
    adj_ht = adj_ht.to(features.device)

    # score_hm = utils.homo_score(labels, adj_hm)
    # print("score_hm: ", score_hm)
    return adj_hm, adj_ht


def con_graph(features, labels, nnodes):

    method = 'heat'
    if method == 'heat':
        dist = -0.5 * pair(features.detach().cpu()) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        dist = F.cosine_similarity(features.detach().unsqueeze(1), features.detach().unsqueeze(0), dim=-1).cpu().numpy()

    weights_lp, weights_hp = torch.zeros((nnodes, nnodes)), torch.zeros((nnodes, nnodes))
    idx_hm, idx_ht = [], []

    k1 = top_k
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i, :], -(k1 + 1))[-(k1 + 1):]
        idx_hm.append(idx)

    counter_hm = 0
    edges_hm = 0

    for i, v in enumerate(idx_hm):
        for Nv in v:
            if Nv == i:
                pass
            else:
                weights_lp[i][Nv] = 1
                if weights_lp[Nv][i] == 0:
                    edges_hm += 1
                if weights_lp[Nv][i] == 0 and labels[Nv] != labels[i]:
                    counter_hm += 1

    k2 = top_k
    for i in range(dist.shape[0]):
        idx = np.argpartition(dist[i, :], k2)[:k2]
        idx_ht.append(idx)

    counter_ht = 0
    edges_ht = 0
    for i, v in enumerate(idx_ht):
        for Nv in v:
            if Nv == i:
                pass
            else:
                weights_hp[i][Nv] = 1
                if weights_hp[Nv][i] == 0:
                    edges_ht += 1
                if weights_hp[Nv][i] == 0 and labels[Nv] == labels[i]:
                    counter_ht += 1

    ht_error = counter_ht / edges_ht
    hm_error = counter_hm /edges_hm
    print("  hm_error: ", hm_error)
    print("  ht_error: ", ht_error)
    print()
    return weights_lp, weights_hp


def build_adjacency_matrix(adj, is_mutual_matrix):
    if is_mutual_matrix:
        new_adj = adj * adj.T
    else:
        new_adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return new_adj

