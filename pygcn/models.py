import torch.nn as nn
import torch
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nemb, nhid, dropout):
        super(GCN, self).__init__()
        self.user_embedding = nn.Embedding(nfeat, nemb)
        self.gc1 = GraphConvolution(nemb, nhid)
        self.gc2 = GraphConvolution(nhid, 128)
        self.linear1 = torch.nn.Linear(128, 16)
        self.linear2 = torch.nn.Linear(16,1)
        self.dropout = dropout

    def forward(self, features, adj):
        user_emb = self.user_embedding(features)
        gcn1 = self.gc1(user_emb,adj)
        x = F.relu(gcn1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.linear1(x)
        x = self.linear2(x)
        return x,user_emb
