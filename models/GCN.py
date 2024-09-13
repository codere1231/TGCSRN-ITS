#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from config.config import *
import torch.nn.functional as F


class mixpropGCN1(nn.Module):

    def __init__(self, c_in, c_out, gdep, dropout, alpha=0.3):
        super(mixpropGCN1, self).__init__()
        self.nconv = gconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout

        self.alpha = alpha

    def forward(self, x, norm_adj, graph_shape=2):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.nconv(norm_adj, h, graph_shape)
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class linear(nn.Module):

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out)

    def forward(self, x):
        return F.relu(self.mlp(x), inplace=True)


class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x, graph_shape=2):
        if graph_shape == 2:
            x = torch.einsum('hw, bwtc->bhtc', (A, x))
        else:
            x = torch.einsum('bhw, bwtc->bhtc', (A, x))
        return x.contiguous()


class GCN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, gdep, dropout):
        super(GCN, self).__init__()
        self.conv1 = mixpropGCN1(c_in, c_hidden, gdep, dropout)
        self.conv2 = mixpropGCN1(c_hidden, c_out, gdep, dropout)

    def forward(self, x, adj, graph_shape=2):
        x1 = self.conv1(x, adj, graph_shape)
        x2 = self.conv2(x1, adj, graph_shape)
        return x2
