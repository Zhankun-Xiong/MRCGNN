import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):

        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, 64 * 64)
        # self.mlp = nn.ModuleList([nn.Linear(16, 64),
        #                           nn.Linear(64, 128),
        #                           nn.Linear(128, 65)
        #                           ])
        self.mlp = nn.ModuleList([nn.Linear(512, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 86)
                                  ])
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def MLP(self, vectors, layer):
        for i in range(layer):
            #vectors = torch.leaky_relu(self.mlp[i](vectors))#
            vectors = self.mlp[i](vectors)
            #print(vectors)ss
        return vectors
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = rels.view(-1, 64, 64)
        heads=heads.view(-1,256)
        tails=tails.view(-1,256)
        scores=torch.cat((heads, tails), dim=1)
        # print(heads.shape)
        # print(tails.shape)
        # print(rels.shape)
        #scores = heads @ rels @ tails.transpose(-2, -1)
        #scores = heads @ tails.transpose(-2, -1)

        # if alpha_scores is not None:
        #  scores = alpha_scores * scores

        #print(scores.shape)
        #scores=scores.view((-1,16))


        scores=self.MLP(scores,7)
        #scores = scores.sum(dim=(-2, -1))
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

