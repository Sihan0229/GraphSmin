import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import utils
import data_load
import random
import copy
from calcu_graph import construct_graph

from args import get_parser
parser = get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(epoch, sorted_indices=None):
    gcl_model.train()
    classifier.train()
    decoder.train()

    optimizer_gcl.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    # augmentation
    features_1, adj_1, features_2, adj_2 = utils.augmentation(features, lp_adj, features, hp_adj, gcl_model.training)

    loss_gcl = gcl_model.forward(features_1, adj_1, features_2, adj_2)
    embed = gcl_model.get_embedding(features_1, features_2, hm_adj, ht_adj, args.B)

    loss_edge = None
    acc_edge = None
    if args.setting == 'new_Graph':
        ori_num = labels.shape[0]
        if args.split_method == 'LT_sampling':
            # long-tailed sampling
            embed, labels_new, idx_train_new, adj_up = utils.long_tailed_sampling(embed, labels, idx_train,
                                                                                  adj=adj.detach(),
                                                                                  portion=args.up_scale,
                                                                                  sorted_indices=sorted_indices[
                                                                                                 : args.K])
        else:
            # step sampling
            embed, labels_new, idx_train_new, adj_up = utils.step_sample(embed, labels, idx_train,
                                                                         adj=adj.detach(),
                                                                         portion=args.up_scale, num_minority=args.K)

        generated_G = decoder(embed)

        loss_edge = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach())

        adj_new = copy.deepcopy(generated_G.detach())
        threshold = 0.5
        adj_new[adj_new < threshold] = 0.0
        adj_new[adj_new >= threshold] = 1.0
        acc_edge = adj_new[:ori_num, :ori_num].eq(adj).double().sum() / (ori_num ** 2)

        adj_new = torch.mul(adj_up, adj_new)
        adj_new[:ori_num, :][:, :ori_num] = adj.detach()
        adj_new = adj_new.detach()

    elif args.setting == 'embed_smote':

        if args.split_method == 'LT_sampling':
            # long-tailed sampling
            embed, labels_new, idx_train_new = utils.long_tailed_sampling(embed, labels, idx_train,
                                                                          adj=None,
                                                                          portion=args.up_scale,
                                                                          sorted_indices=sorted_indices[: args.K])
        else:
            # step sampling
            embed, labels_new, idx_train_new = utils.step_sample(embed, labels, idx_train, adj=None,
                                                                 portion=args.up_scale, num_minority=args.K)

        adj_new = adj
    else:
        labels_new = labels
        idx_train_new = idx_train
        adj_new = adj

    output = classifier(embed, adj_new)

    loss_cls = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])
    acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])

    if args.setting == 'new_Graph':
        loss = loss_cls + loss_edge * 0.001 + loss_gcl
    elif args.setting == 'embed_smote':
        loss = loss_cls + loss_gcl
        loss_edge = loss_cls + loss_gcl
    else:
        loss = loss_cls
        loss_edge = loss_cls

    loss.backward()
    optimizer_gcl.step()
    optimizer_cls.step()

    if args.setting == 'new_Graph':
        optimizer_de.step()

    if args.setting == 'new_Graph':
        print('Epoch: {:05d}'.format(epoch + 1),
              'loss_cls: {:.4f}'.format(loss_cls.item()),
              'loss_edge: {:.4f}'.format(loss_edge.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_edge: {:.4f}'.format(acc_edge.item()))
    else:
        print('Epoch: {:05d}'.format(epoch + 1),
              'loss_cls: {:.4f}'.format(loss_cls.item()),
              'loss_edge: {:.4f}'.format(loss_edge.item()),
              'acc_train: {:.4f}'.format(acc_train.item()))


def test():
    gcl_model.eval()
    classifier.eval()
    decoder.eval()

    features_1, adj_1, features_2, adj_2 = utils.augmentation(features, lp_adj, features, hp_adj, gcl_model.training)
    embed = gcl_model.get_embedding(features_1, features_2, hm_adj, ht_adj, args.B)

    output = classifier(embed, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])

    print("current test results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()),
          end=" ")

    auc_score, macro_F = utils.print_class_acc(output[idx_test], labels[idx_test])

    return acc_test, auc_score, macro_F


if __name__ == '__main__':

    t_total = time.time()

    # Load data
    features, labels = data_load.get_info_from_file()

    hm_adj, ht_adj = construct_graph(features, labels)
    adj = hm_adj.clone()
    input_dim = len(features[0])

    adj += torch.eye(adj.shape[0])
    lp_adj = utils.normalize_lp_adj(hm_adj)
    hp_adj = utils.normalize_hp_adj(ht_adj)

    setup_seed(args.seed)
    sorted_indices = []
    if args.split_method == 'LT_sampling':
        idx_train, idx_test, sorted_indices = utils.split_data(labels, args.IR, args.K)
    else:
        idx_train, idx_test = utils.split_data(labels, args.IR, args.K)

    # Model and optimizer
    if args.setting == 'new_Graph' or args.setting == 'gnn':
        if args.model == 'sage':
            classifier = models.Sage_Classifier(nembed=args.embed_dim,
                                                nhid=args.hid_dim,
                                                nclass=labels.max().item() + 1,
                                                dropout=args.dropout)
        else:
            classifier = models.GCN_Classifier(nembed=args.embed_dim,
                                               nhid=args.hid_dim,
                                               nclass=labels.max().item() + 1,
                                               dropout=args.dropout)
    else:
        classifier = models.Classifier(nembed=args.embed_dim,
                                       nhid=args.hid_dim,
                                       nclass=labels.max().item() + 1,
                                       dropout=args.dropout)

    gcl_model = models.GCL(input_dim, args.embed_dim, args.proj_dim, args)
    decoder = models.Decoder(nembed=128, dropout=args.dropout)

    optimizer_gcl = torch.optim.Adam(gcl_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        gcl_model.to(device)
        gcl_model.encoder1.to(device)
        gcl_model.encoder2.to(device)

        classifier = classifier.to(device)
        decoder = decoder.to(device)
        features = features.to(device)
        adj = adj.to(device)
        hm_adj = hm_adj.to(device)
        ht_adj = ht_adj.to(device)
        lp_adj = lp_adj.to(device)
        hp_adj = hp_adj.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_test = idx_test.to(device)

    best_acc, best_auc, best_F = 0, 0, 0

    # if select hlp contrastive learning method, you need to process this row
    gcl_model.set_mask(hm_adj)

    # training process
    sub_time = time.time()
    for index in range(args.epochs):
        train(index, sorted_indices)
        if index % 10 == 0:
            t_acc, auc_score, macro_F = test()
            if macro_F > best_F:
                best_F = macro_F
                best_auc = auc_score
                best_acc = t_acc
    print()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("ACC: {:.4f}".format(best_acc))
    print("AUC-ROC score: {:.4f}".format(best_auc),
          "F1-score: {:.4f}".format(best_F))


