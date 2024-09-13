# !/usr/bin/env python
# -*- coding:utf-8 -*-


from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import pandas as pd
import math
from scipy.sparse import linalg
from torch.optim.lr_scheduler import MultiStepLR
import colorsys
import random


class StepLR2(MultiStepLR):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):

        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxNormalization:
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        data = 1. * (data - self.min) / (self.max - self.min)
        data = 2. * data - 1.
        return data

    def inverse_transform(self, data):
        data = (data + 1) / 2
        data = data * (self.max - self.min) + self.min
        return data


class StandardScaler_Torch:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_norm_Adj(W):
    assert W.shape[0] == W.shape[1]
    N = W.shape[0]
    D = np.sum(W, axis=1)
    D = np.diag(D ** -0.5)
    sym_norm_Adj_matrix = np.dot(D, W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, D)
    return sym_norm_Adj_matrix


def asym_norm_Adj(W):
    assert W.shape[0] == W.shape[1]
    N = W.shape[0]
    D = np.diag(1.0 / np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)
    return norm_Adj_matrix


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sparse.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def kl_normal_log(mu, logvar, mu_prior, logvar_prior):
    var = logvar.exp()
    var_prior = logvar_prior.exp()

    element_wise = 0.5 * (
            torch.log(var_prior) - torch.log(var) + var / var_prior + (mu - mu_prior).pow(2) / var_prior - 1)
    kl = element_wise.mean(-1)

    kl = torch.mean(kl, dim=1)
    kl = torch.mean(kl, dim=0)

    return kl


def kl_normal(mu, var, mu_prior, var_prior):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        mu: tensor: (batch, node, dim): 后验 mean
        var: tensor: (batch, node, dim): 后验 variance
        mu_prior: tensor: (batch, node, dim): 先验 mean
        var_prior: tensor: (batch, node, dim): 先验 variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """

    element_wise = 0.5 * (
            torch.log(var_prior) - torch.log(var) + var / var_prior + (mu - mu_prior).pow(2) / var_prior - 1)
    kl = element_wise.sum(-1)

    kl = torch.mean(kl, dim=1)
    kl = torch.mean(kl, dim=0)
    return kl


def prune(A):
    zero = torch.zeros_like(A).to(A.device)
    A = torch.where(A < 0.3, zero, A)
    return A


def gumble_dag_loss(A):
    expm_A = torch.exp(F.gumbel_softmax(A))
    l = torch.trace(expm_A) - A.size()[0]
    return l


def matrix_poly(matrix, d):
    x = torch.eye(d).to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def stau(w, tau):
    prox_plus = torch.nn.Threshold(0., 0.)
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss += 2 * tol - torch.sum(torch.abs(A[:, i])) - torch.sum(torch.abs(A[i, :])) + z * z
    return loss


# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):
    result = - A + z_positive * z_positive
    loss = torch.sum(result)

    return loss


def make_saved_dir(saved_dir, use_time=3):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if use_time == 1:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d_%H:%M'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
    elif use_time == 2:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

    return saved_dir


def get_timestamp(demand: pd.DataFrame, holidays, start_date, end_date, freq):
    time_date = pd.date_range(start_date, end_date, freq=freq, closed='left')

    time = pd.DataFrame(index=time_date)
    time['dayofweek'] = time_date.weekday
    time['timeofday'] = (time_date.hour * 3600 + time_date.minute * 60 + time_date.second) // (int(freq[:-3]) * 60)
    time['holidays'] = time.index.strftime('%Y-%m-%d').isin(holidays).astype(int)

    return time


def gumbel_sigmoid(logits, temperature=0.1, noise=False, hard=False, mode='sigmoid'):
    if noise:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        logits = logits + gumbel_noise

    if mode == 'sigmoid':
        y_soft = torch.sigmoid(logits / temperature)
    elif mode == 'tanh':
        y_soft = torch.tanh(logits)

    if hard:
        y_hard = torch.where(y_soft > 0.5, 1, 0)
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft

    return y


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def gen_color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)
