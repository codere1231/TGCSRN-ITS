# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime
from config.config import *
from models.GCN import GCN
from tools.utils import _h_A, gumbel_sigmoid
import math
from models.TCN import get_tcn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TGCSRN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_nodes, geo_graph, od_graph, region_belong,
                 context_channels, context_dims, activation_type, cluster_num, device, type, gdep, dropout, fuse_type,
                 use_skip, num_for_predict, use_weather, use_adaptive, node_emb, d_model, num_for_target,
                 num_hours_per_day, blocks):
        super().__init__()
        self.device = device
        self.cluster_num = cluster_num
        self.region_belong = region_belong
        self.nb_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.geo_graph = geo_graph
        self.od_graph = od_graph
        self.use_weather = use_weather
        self.use_adaptive = use_adaptive
        self.node_emb = node_emb
        self.num_hours_per_day = num_hours_per_day
        self.fuse_type = fuse_type
        self.use_skip = use_skip
        self.tpos_channel = context_channels[0]
        self.weather_channel = context_channels[1]
        self.t_emb_dim = hidden_dim
        self.tpos_dim = context_dims[0]
        self.weather_dim = context_dims[1]
        self.num_for_target = num_for_target
        self.num_tcn_t_now = num_for_predict - 1
        self.num_tcn_t_pre = num_for_predict - 2
        self.num_tcn_k = num_for_predict - 1
        self.d_model = d_model
        self.scale = 1 / math.sqrt(self.d_model)
        self.envCodebook = nn.Parameter(torch.randn(cluster_num, d_model).to(device), requires_grad=True)
        self.nodesCodebook = nn.Parameter(torch.randn(num_nodes, d_model).to(device), requires_grad=True)
        self.w_v = nn.Parameter(torch.randn(d_model, cluster_num).to(device), requires_grad=True)
        self.w_k = nn.Parameter(torch.randn(d_model, d_model).to(device), requires_grad=True)
        self.w_q = nn.Parameter(torch.randn(d_model, d_model).to(device), requires_grad=True)
        self.softmax_mat = nn.Softmax(dim=1)
        self.core_fc = nn.Linear(self.hidden_dim, self.hidden_dim ** 2)
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        self.num_for_predict = num_for_predict
        self.fc_x_t = nn.Linear(self.in_dim, self.hidden_dim)
        self.tcns = nn.ModuleList([get_tcn(in_dim=2 * self.hidden_dim,
                                           out_dim=self.hidden_dim,
                                           residual_channels=self.hidden_dim * 2,
                                           dilation_channels=self.hidden_dim * 2,
                                           num_nodes=self.nb_nodes,
                                           blocks=blocks)
                                   for _ in range(self.cluster_num)])
        self.tcns_prev = nn.ModuleList([get_tcn(in_dim=2 * self.hidden_dim,
                                                out_dim=self.hidden_dim,
                                                residual_channels=self.hidden_dim * 2,
                                                dilation_channels=self.hidden_dim * 2,
                                                num_nodes=self.nb_nodes,
                                                blocks=blocks)
                                        for _ in range(self.cluster_num)])

        if self.geo_graph == None:
            self.gcnlayers1 = None
        else:
            self.gcnlayers1 = GCN(hidden_dim, hidden_dim, hidden_dim, gdep, dropout)

        self.gcnlayers2 = GCN(hidden_dim, hidden_dim, hidden_dim, gdep, dropout)

        self.bn_1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn_1_prev = nn.BatchNorm1d(self.hidden_dim)

        self.nodevec_p1 = nn.Parameter(torch.randn(self.num_hours_per_day * 2, self.hidden_dim).to(device),
                                       requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(self.nb_nodes, self.hidden_dim).to(device),
                                       requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(self.nb_nodes, self.hidden_dim).to(device),
                                       requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, self.hidden_dim).to(device),
                                       requires_grad=True).to(device)
        self.gconv_p = GCN(hidden_dim, hidden_dim, hidden_dim, gdep, dropout)

        self.fc_out = nn.ModuleList([MLP(self.hidden_dim, 100, self.out_dim) for _ in range(num_for_target)])

    def init_soft_mat(self):
        with torch.no_grad():
            for i in range(self.cluster_num):
                self.soft_mat[self.region_belong[i], i] = 1

    def dgconstruct_2(self, time_embedding, source_embedding, target_embedding, core_embedding):
        if time_embedding.dim() == 1:
            time_embedding = time_embedding.unsqueeze(0)
        if source_embedding.dim() == 1:
            source_embedding = source_embedding.unsqueeze(0)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, t_pos, weather, ind, pre_training_mat=None, mat_mode='train'):
        batch_size, node, time_slot, in_dim = x.shape
        ind = torch.where(ind[:, :, 0] <= 4, ind[:, :, 1], self.num_hours_per_day - 1 + ind[:, :, 1]).squeeze()
        total_t_pos = torch.where(t_pos[:, :, 0] <= 4, t_pos[:, :, 1], self.num_hours_per_day - 1 + t_pos[:, :, 1])
        t_emb = self.timestep_embedding(total_t_pos.flatten(), self.t_emb_dim).reshape(
            (batch_size, time_slot, self.t_emb_dim))
        t_emb_reshape = t_emb.unsqueeze(1).repeat([1, self.nb_nodes, 1, 1])
        cur_h = torch.zeros(batch_size, self.nb_nodes, self.hidden_dim).to(self.device)

        if mat_mode == 'train':
            mat_value = torch.matmul(self.envCodebook, self.w_v)
            mat_key = torch.matmul(self.envCodebook, self.w_k)
            mat_query = torch.matmul(self.nodesCodebook, self.w_q)
            scores = torch.einsum('nd,kd->nk', mat_query, mat_key)
            scores *= self.scale
            gumbel_mat = self.softmax_mat(scores)
            gumbel_mat = torch.einsum('nk,kl->nl', gumbel_mat, mat_value)
            gumbel_mat = F.gumbel_softmax(gumbel_mat, tau=1, hard=True, dim=1)
        elif mat_mode == 'test_only':
            gumbel_mat = pre_training_mat

        expand_mask_x = gumbel_mat.unsqueeze(-1).expand(batch_size, self.nb_nodes, self.cluster_num,
                                                        2 * self.hidden_dim)

        unbias_rep_t = torch.zeros_like(cur_h)
        h_prev = torch.zeros_like(cur_h)
        x_t = x[:, :, :, :]
        t_emb_reshape_t = t_emb_reshape[:, :, :, :]
        x_t_emb = self.fc_x_t(x_t)
        total_x_t = torch.cat([x_t_emb, t_emb_reshape_t], dim=-1)

        for i in range(self.cluster_num):
            x_tmp = total_x_t * (expand_mask_x[:, :, i, :].squeeze().unsqueeze(2).repeat([1, 1, self.num_tcn_k + 1, 1]))
            x_flit = torch.index_select(x_tmp, 1, torch.nonzero(expand_mask_x[0, :, i, 0]).squeeze())
            x_flit_tem = torch.nonzero(expand_mask_x[0, :, i, 0])
            if not x_flit_tem.squeeze().numel():
                continue

            h_flit_1 = self.tcns[i](x_flit[:, :, 1:, :], self.num_tcn_k)
            h_flit_2 = h_flit_1[:, :, 0, :].squeeze()
            h_flit = h_flit_2.view(batch_size, -1, self.hidden_dim)
            if len(x_flit_tem) == 1:
                unbias_rep_t[:, x_flit_tem.squeeze(), :] = h_flit.squeeze()
            else:
                unbias_rep_t[:, x_flit_tem.squeeze(), :] = h_flit

            h_flit_1_prev = self.tcns_prev[i](x_flit[:, :, :-1, :], self.num_tcn_k)
            h_flit_2_prev = h_flit_1_prev[:, :, 0, :].squeeze()
            h_flit_prev = h_flit_2_prev.view(batch_size, -1, self.hidden_dim)
            if len(x_flit_tem) == 1:
                h_prev[:, x_flit_tem.squeeze(), :] = h_flit_prev.squeeze()
            else:
                h_prev[:, x_flit_tem.squeeze(), :] = h_flit_prev

        unbias_rep_t = unbias_rep_t.permute((0, 2, 1))
        unbias_rep_t = self.bn_1(unbias_rep_t)
        unbias_rep_t = unbias_rep_t.permute((0, 2, 1))

        h_prev = h_prev.permute((0, 2, 1))
        h_prev = self.bn_1_prev(h_prev)
        h_prev = h_prev.permute((0, 2, 1))

        if self.gcnlayers1 is None:
            h1 = torch.zeros_like(unbias_rep_t)
        else:
            h1 = self.gcnlayers1(h_prev.unsqueeze(2), self.geo_graph).squeeze(2)

        core_t = t_emb[:, self.num_tcn_t_now, :]
        core_t = self.core_fc(core_t).reshape(batch_size, self.hidden_dim, self.hidden_dim)
        causal_map = torch.bmm(unbias_rep_t, core_t)
        causal_map = torch.bmm(causal_map, h_prev.transpose(-1, -2))
        causal_map = F.softmax(causal_map, dim=-1)
        h2 = self.gcnlayers2(unbias_rep_t.unsqueeze(2), causal_map, graph_shape=3).squeeze(2)

        adp = self.dgconstruct_2(self.nodevec_p1[ind[:, -self.num_tcn_t_now]], self.nodevec_p2, self.nodevec_p3,
                                 self.nodevec_pk)
        h3 = self.gconv_p(unbias_rep_t.unsqueeze(2), adp, graph_shape=3).squeeze(2)

        cur_h = h1 + h2 + h3

        # output layer
        output = torch.zeros(batch_size, self.nb_nodes, self.num_for_target, self.out_dim).to(self.device)
        for i in range(self.num_for_target):
            output[:, :, i, :] = self.fc_out[i](cur_h)
        return output, gumbel_mat

def get_TGCSRN(in_dim=2,
               out_dim=1,
               hidden_dim=32,
               num_for_target=1,
               num_for_predict=12,
               num_nodes=54,
               context_channels=None,
               context_dims=None,
               activation_type='relu',
               supports=None,
               region_belong=None,
               cluster_num=7,
               device='cuda:1',
               gcn_depth=1,
               fuse_type='sum',
               use_skip=False,
               use_weather=False,
               use_adaptive=False,
               node_emb=10,
               d_model=64,
               num_hours_per_day=288,
               blocks=4):
    if supports == None:
        geo_graph = None
        od_graph = None
    else:
        geo_graph = supports[0]
        od_graph = None
    model = TGCSRN(in_dim, out_dim, hidden_dim, num_nodes, geo_graph, od_graph, region_belong,
                   context_channels, context_dims, activation_type, cluster_num, device, 'grus', gcn_depth, 0.3,
                   fuse_type, use_skip, num_for_predict, use_weather, use_adaptive, node_emb, d_model,
                   num_for_target=num_for_target, num_hours_per_day=num_hours_per_day, blocks=blocks)
    return model
