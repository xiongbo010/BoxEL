#!/usr/bin/env python
# coding: utf-8

from scipy.stats import rankdata
import statistics
from collections import Counter
import torch 
from basic_box import Box
import numpy as np

def compute_cond_probs(model, boxes1, boxes2):
    log_intersection = torch.log(torch.clamp(model.volumes(model.intersection(boxes1, boxes2)), 1e-10, 1e4))
    log_box2 = torch.log(torch.clamp(model.volumes(boxes2), 1e-10, 1e4))
    return torch.exp(log_intersection-log_box2)



def compute_rank_roc(ranks, n):
    auc_lst = list(ranks.keys())
    auc_x = auc_lst[1:]
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x)/n
    return auc

def compute_accuracy(model, test_data):
    nf1_min = model.min_embedding[test_data[:,[0,2]]]
    nf1_delta = model.delta_embedding[test_data[:,[0,2]]]
    nf1_max = nf1_min+torch.exp(nf1_delta)
    boxes1 = Box(nf1_min[:, 0, :], nf1_max[:, 0, :])
    boxes2 = Box(nf1_min[:, 1, :], nf1_max[:, 1, :])
    probs = compute_cond_probs(model, boxes1, boxes2).cpu().detach().numpy()
    return np.sum(probs==1)/probs.shape[0]








