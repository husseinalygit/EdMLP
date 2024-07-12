#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/06/28 17:51
@Author: Merc2
'''
import numpy as np
from pathlib import Path
import openml
import torch
from torch.utils.data import Dataset

class OPML(Dataset):
    def __init__(self, task=None, cv=0, train=True):# volkert 

        # _, self.CV_NUM, _ = task.get_split_dimensions()
        self.N_TYPES = len(task.class_labels)

        X,y = task.get_X_and_y()
        X,y = torch.tensor(X), torch.tensor(y)
        X[torch.isnan(X)] = 0.

        X = X[:,~((X == 0).sum(0) == X.shape[0])]

        X = (X-X.mean(0))/X.std(0)

        self.cv = cv
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=cv, sample=0)
        if train is True:
            self.X = X[train_indices]
            self.y = y[train_indices]
        else:
            self.X = X[test_indices]
            self.y = y[test_indices]
        self.SEQ = X.shape[1]
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index].float()
        y = self.y[index].long()
        return X,y

