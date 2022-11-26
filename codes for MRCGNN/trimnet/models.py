import torch
#from torch import nn
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import (GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool)

from layers import (
    CoAttentionLayer,
    RESCAL,
    RESCAL
)
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing
import math
from torch.nn import Linear, GRU, Parameter
from torch.nn.functional import leaky_relu
from torch_geometric.nn import Set2Set, NNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.init import kaiming_uniform_, zeros_

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class MultiHeadTripletAttention(MessagePassing):
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(MultiHeadTripletAttention, self).__init__(aggr='add', node_dim=0, **kwargs)  # aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)  # time consuming 12.14s
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        # return x_j * alpha
        # return self.prelu(alpha * e_ij * x_j)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)


class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for i in range(self.time_step):
            m = F.celu(self.conv.forward(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.ln(x.squeeze(0))
        return x


class TrimNet(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, depth=3, heads=4, dropout=0.1, outdim=86,kge_dim=128, rel_total=86):
        super(TrimNet, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.initial_norm = LayerNorm(55)
        self.lin0 = Linear(55, hidden_dim)
        self.lin1 = Linear(128, 65)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.convs = nn.ModuleList([
            Block(hidden_dim, edge_in_dim, heads)
            for i in range(depth)
        ])
        self.dropout1 = torch.nn.Dropout(p=0.1)
        #self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=3)

        self.mlp = nn.ModuleList([nn.Linear(256, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 65)
                                  ])


    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors
    def forward(self, triples):
        h_data, t_data, rels = triples
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        x = F.celu(self.lin0(h_data.x))
        x1 = F.celu(self.lin0(t_data.x))
        for conv in self.convs:
            x = x + F.dropout(conv(x, h_data.edge_index, h_data.edge_attr), p=self.dropout, training=self.training)
        x = self.set2set(x, h_data.batch)
        x=F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x1 = x1 + F.dropout(conv(x1, t_data.edge_index, t_data.edge_attr), p=self.dropout, training=self.training)
        x1 = self.set2set(x1, t_data.batch)
        x1=F.dropout(x1, p=self.dropout, training=self.training)

        xall=torch.cat((x,x1),1)



        scores=self.MLP(xall,7)

        return scores,rels


    def get_weight(self, triples,zhongzi):
        #print(triples)
        h_data, _, _= triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        x = F.celu(self.lin0(h_data.x))
        for conv in self.convs:
            x = x + F.dropout(conv(x, h_data.edge_index, h_data.edge_attr), p=self.dropout, training=self.training)
        x = self.set2set(x, h_data.batch)
        xall = x#self.out(x)

        #repr_h = torch.stack(xall,dim=-2)
        repr_h=xall.view((572,-1))
        #print(repr_h.shape)
        np.save('drug_emb_trimnet'+str(zhongzi)+'.npy',repr_h.cpu())

        kge_heads = repr_h


        return kge_heads

