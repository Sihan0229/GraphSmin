import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import Sequential, Linear, ReLU


class GraphConvolution(Module):
    """
    图卷积网络（GCN）的基本层
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #for 3_D batch, need a loop!!!


        if self.bias is not None:
            return output + self.bias
        else:
            return output


#Graphsage layer
class SageConv(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()
        self.proj = nn.Linear(in_features*2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        if adj.layout != torch.sparse_coo:
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1],-1))+1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1)+1)
        else:
            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1)+1)

        data = torch.cat([features,neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


class Sage_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(nfeat, nembed)
        self.fc2 = nn.Linear(nhid, nembed)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.fc2(x))
        return x


class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x


class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)
        return x

# 边预测器（Edge Predictor），用于生成新节点与原始节点之间的连接
class Decoder(Module):
    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))
        return adj_out

# 计算两个嵌入向量之间的相似性，用于对比学习中的损失计算。
# 它通过归一化嵌入向量并计算它们的点积来实现。
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


class GCL(nn.Module):
    def __init__(self, input_dim, embedding_dim, proj_dim, args, tau: float = 0.2):
        super(GCL, self).__init__()
        self.neg_mask = None
        self.pos_mask = None

        if args.model == 'gcn':
            self.encoder1 = GCN_En(input_dim, embedding_dim, embedding_dim, args.dropout)
            self.encoder2 = GCN_En(input_dim, embedding_dim, embedding_dim, args.dropout)
        else:
            self.encoder1 = Sage_En(input_dim, embedding_dim, embedding_dim, args.dropout)
            self.encoder2 = Sage_En(input_dim, embedding_dim, embedding_dim, args.dropout)

        self.tau: float = tau

        self.proj_head1 = Sequential(
            Linear(embedding_dim, proj_dim),
            ReLU(inplace=True),
            Linear(proj_dim, proj_dim)
        )
        self.proj_head2 = Sequential(
            Linear(embedding_dim, proj_dim),
            ReLU(inplace=True),
            Linear(proj_dim, proj_dim)
        )

    def forward(self, feature1, adj1, feature2, adj2):
        embed1 = self.encoder1(feature1, adj1)
        embed2 = self.encoder2(feature2, adj2)

        proj1 = self.proj_head1(embed1)
        proj2 = self.proj_head2(embed2)
        # return self.normal_loss(proj1, proj2)
        return self.lhp_loss(proj1, proj2)

    # Find difficult sample pairs
    def set_mask(self, hm_adj):
        num = len(hm_adj[0])
        pos_mask = hm_adj.detach()
        device = pos_mask.device
        neg_mask = (torch.ones(num, num, device=device) - pos_mask).detach()

        self.pos_mask = pos_mask
        self.neg_mask = neg_mask

    def lhp_loss(self, proj1, proj2):
        loss_0 = self.inforce(proj1, proj2)
        loss_1 = self.inforce(proj2, proj1)
        loss = (loss_0 + loss_1) / 2.0
        return loss.mean()

    def inforce(self, z1: torch.Tensor, z2: torch.Tensor, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        f = lambda x: torch.exp(x / 0.2)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))

        between_pos_mask_up = torch.mul(between_sim, pos_mask).sum(1)
        between_neg_mask_low = torch.mul(between_sim, neg_mask).sum(1)
        refl_neg_mask_low = torch.mul(refl_sim, neg_mask).sum(1)
        return -torch.log(between_pos_mask_up / (between_neg_mask_low + refl_neg_mask_low))

    # from GCA
    def normal_loss(self, proj1, proj2):
        loss_0 = self.nec_loss(proj1, proj2)
        loss_1 = self.nec_loss(proj2, proj1)
        loss = (loss_0 + loss_1) / 2.0
        return loss.mean()

    def nec_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def get_embedding(self, feature1, feature2, adj1, adj2, β):
        embed1 = self.encoder1(feature1, adj1)
        embed2 = self.encoder2(feature2, adj2)
        return β*embed1 + (1-β)*embed2

    def get_embedding2(self, feature1, feature2, adj1, adj2):
        embed1 = self.encoder1(feature1, adj1)
        embed2 = self.encoder2(feature2, adj2)
        return torch.cat((embed1, embed2), dim=1)




