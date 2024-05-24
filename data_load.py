import scipy.sparse as sp
import numpy as np
import torch

from args import get_parser
parser = get_parser()
args = parser.parse_args()


def get_info_from_file():
    """get data from file"""
    data = np.loadtxt('./data/transformer/github.csv', dtype=float)
    data = np.unique(data, axis=0)
    features = data[:, :-1]
    labels = data[:, -1]
    features = normalize(features)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels-1)
    # print("f shape: ", features.shape)
    # print("l shape: ", labels.shape)

    DGA2000_fault_types = ['Normal', 'Low-Temperature Overheating', 'Medium-Temperature Overheating',
                           'High Temperature Overheating',
                           'Partial Discharge', 'Low-Energy Discharge', 'High-Energy Discharge']
    C = len(set(labels.tolist()))
    num_classes = [sum(map(lambda x: x == i, labels.numpy())) for i in range(C)]
    print("*********************************************")
    print("************Fault Data Statistics************")
    print("*********************************************")
    print("c_index num types")
    for i in range(len(DGA2000_fault_types)):
        print("{:d} {:9d} {}".format(i, num_classes[i], DGA2000_fault_types[i]))
    print("*********************************************")
    print()
    print("*********************************************")
    print("setting:", args.setting)
    print("basic model:", args.model)
    print("sampling method:", args.split_method)
    print("seed:", args.seed)
    print("Imb ratio:", args.IR)
    print("up_scale ratio:", args.up_scale)
    if args.split_method == 'LT_sampling':
        print("num_major: ", args.LT_major)
    else:
        print("num_major: ", args.ST_major)
    print("num_SMOTE_classes:", args.K)
    print("K_neighbors:", args.topK)
    print("epochs:", args.epochs)
    print("*********************************************")
    print()

    return features, labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
