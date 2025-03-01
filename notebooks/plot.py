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

def plot(loss, title, losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("# epoch")
    plt.ylabel(loss)
    plt.title(title)
    plt.savefig('/Shared/EvoGraphNet/plots/' + title + '.png')
    plt.close()

# COMMAND ----------

def plot_matrix(out, fold, sample, epoch, strategy):
    fig = plt.figure()
    plt.pcolor(abs(out))
    plt.colorbar()
    plt.imshow(out)
    title = "Generator Output, Epoch = " + str(epoch) + " Fold = " + str(fold) + " Strategy = " + strategy
    plt.title(title)
    plt.savefig('/Shared/EvoGraphNet/plots/' + str(fold) + 'Gen_' + str(sample) + '_' + str(epoch) + '.png')
    plt.close()        


        
                
                
