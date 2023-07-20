# Databricks notebook source
import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable

# COMMAND ----------

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# COMMAND ----------

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

#############################################
n_roi = 155 #### number of ROIs
###########################################

# COMMAND ----------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        lin = Sequential(Linear(1, n_roi*n_roi), ReLU())
        self.conv1 = NNConv(n_roi, n_roi, lin, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(n_roi, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
        lin = Sequential(Linear(1, n_roi), ReLU())
        self.conv2 = NNConv(n_roi, 1, lin, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
        lin = Sequential(Linear(1, n_roi), ReLU())
        self.conv3 = NNConv(1, n_roi, lin, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(n_roi, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x1 = torch.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)
        #Below 2 lines are the corrections
        x1 = (x1 + x1.T) / 2.0
        x1.fill_diagonal_(fill_value = 0)
        x2 = torch.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)
        x3 = torch.cat([torch.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:n_roi]
        x5 = x3[:, n_roi:n_roi+n_roi]
        x6 = (x4 + x5) / 2
        #Below 2 lines are the corrections
        x6 = (x6 + x6.T) / 2.0
        x6.fill_diagonal_(fill_value = 0)
        return x6

# COMMAND ----------

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        lin = Sequential(Linear(2, n_roi*n_roi), ReLU())
        self.conv1 = NNConv(n_roi, n_roi, lin, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(n_roi, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        lin = Sequential(Linear(2, n_roi), ReLU())
        self.conv2 = NNConv(n_roi, 1, lin, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data, data_to_translate):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr_data_to_translate = data_to_translate.edge_attr

        edge_attr_data_to_translate_reshaped = edge_attr_data_to_translate.view(n_roi*n_roi, 1)

        gen_input = torch.cat((edge_attr, edge_attr_data_to_translate_reshaped), -1)
        x = F.relu(self.conv11(self.conv1(x, edge_index, gen_input)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv22(self.conv2(x, edge_index, gen_input)))

        return torch.sigmoid(x)

