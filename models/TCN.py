# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class TCN(nn.Module):
    def __init__(self, device,
                 num_nodes=175,
                 in_dim=128,
                 out_dim=64,
                 residual_channels=64,
                 dilation_channels=64,
                 kernel_size=2,
                 blocks=2,
                 layers=2,
                 flag_double_residual=False,
                 flag_start_conv=False,
                 flag_end_conv=False,
                 flag_double_residual_ReLU=False,
                 flag_double_residual_sigmoid=False,
                 flag_drop=False,
                 drop=0.5):
        super(TCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.blocks = blocks
        self.layers = layers
        self.num_nodes = num_nodes
        self.flag_double_residual = flag_double_residual
        self.flag_start_conv = flag_start_conv
        self.flag_end_conv = flag_end_conv
        self.flag_drop = flag_drop
        self.flag_double_residual_ReLU = flag_double_residual_ReLU
        self.flag_double_residual_sigmoid = flag_double_residual_sigmoid

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        if flag_double_residual_ReLU:
            self.relu = nn.ReLU()
        if flag_drop:
            self.drop = nn.Dropout(drop)
        if flag_start_conv:
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
        if flag_end_conv:
            self.end_conv = nn.Conv2d(in_channels=residual_channels,
                                      out_channels=out_dim,
                                      kernel_size=(1, 1))

        receptive_field = 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.receptive_field = receptive_field

    def forward(self, input, num_known):
        input = input.permute((0, 3, 1, 2))
        in_len = input.size(3)
        if num_known != in_len:
            raise ValueError("The number of known points is not equal to the receptive field!")
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        elif in_len == self.receptive_field:
            x = input
        else:
            raise ValueError("The input sequence length is too long!")

        if self.flag_start_conv:
            x = self.start_conv(x)

        for i in range(self.blocks * self.layers):
            #                 |-- conv -- tanh --|
            # -> dilate -|----|                  *  -->	*next input*
            #                 |-- conv -- sigm --|

            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            if self.flag_drop:
                filter = self.drop(filter)

            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            if self.flag_double_residual:
                x = x + residual[:, :, :, -x.size(3):]
                if self.flag_double_residual_ReLU and self.flag_double_residual_sigmoid:
                    raise ValueError
                if self.flag_double_residual_ReLU:
                    x = self.relu(x)
                if self.flag_double_residual_sigmoid:
                    x = torch.sigmoid(x)

            x = self.bn[i](x)

        if self.flag_end_conv:
            x = self.end_conv(x)
        x = x.permute((0, 2, 3, 1))

        return x


def get_tcn(in_dim=128,
            out_dim=64,
            residual_channels=64,
            dilation_channels=64,
            num_nodes=175,
            device='cuda:1',
            kernel_size=2,
            blocks=2,
            layers=2,
            flag_double_residual=True,
            flag_start_conv=False,
            flag_end_conv=True,
            flag_double_residual_ReLU=False,
            flag_double_residual_sigmoid=False,
            flag_drop=False,
            drop=0.5):
    model = TCN(device, num_nodes=num_nodes, in_dim=in_dim, out_dim=out_dim, residual_channels=residual_channels,
                dilation_channels=dilation_channels, kernel_size=kernel_size, blocks=blocks, layers=layers,
                flag_double_residual=flag_double_residual, flag_start_conv=flag_start_conv, flag_end_conv=flag_end_conv,
                flag_double_residual_ReLU=flag_double_residual_ReLU, flag_drop=flag_drop,
                flag_double_residual_sigmoid=flag_double_residual_sigmoid, drop=drop)

    return model
