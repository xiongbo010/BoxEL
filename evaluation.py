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

# def compute_mean_rank(model, valid_data,classes,device):
#     classes_index = list(classes.values())
#     classes_index = torch.Tensor(classes_index).to(device).reshape(-1,1).long()

#     mean_rank = 0.0
#     n = len(valid_data)
#     for i, (c, r, d) in enumerate(valid_data):
#         c_data = torch.cat((c.repeat(classes_index.shape[0], 1), torch.Tensor([0]).repeat(classes_index.shape[0], 1).to(device).long(), classes_index), 1) 
#         nf1_min = model.min_embedding[c_data[:,[0,2]]]
#         point1 = nf1_min[:, 0, :]
#         point2 = nf1_min[:, 1, :]
#         relation = model.relation_embedding[c_data[:,1]]
#         scaling = model.scaling_embedding[c_data[:,1]]

#         trans_point = point1*scaling+relation
#         role_inclusion_loss = torch.norm(trans_point-points2,p=2, dim=1,keepdim=True)
#         c_probs = role_inclusion_loss.cpu().detach().numpy()
#         index = rankdata(c_probs, method='average')
#         rank = index[d]
#         mean_rank += rank 
#     mean_rank /= n
#     return mean_rank

# def compute_mean_rank(model, valid_data,classes,device):
#     classes_index = list(classes.values())
#     classes_index = torch.Tensor(classes_index).to(device).reshape(-1,1).long()

#     mean_rank = 0.0
#     n = len(valid_data)
#     for i, (c, r, d) in enumerate(valid_data):

#         c_data = torch.cat((c.repeat(classes_index.shape[0], 1), torch.Tensor([0]).repeat(classes_index.shape[0], 1).to(device).long(), classes_index), 1) 
#         protein = model.min_embedding[c_data[:,[0,2]]]
#         points1 = protein[:, 0, :]
#         points2 = protein[:, 1, :]
#         relation = model.relation_embedding[c_data[:,1]]
#         scaling = model.scaling_embedding[c_data[:,1]]
#         trans_point = points1*scaling+relation
#         c_probs = torch.norm(trans_point-points2,p=2, dim=1,keepdim=True)

#         index = rankdata(c_probs, method='average')
#         rank = index[d]
#         mean_rank += rank 
#     mean_rank /= n
#     return mean_rank

def compute_mean_rank(model, valid_data,classes,device):
    classes_index = list(classes.values())
    classes_index = torch.Tensor(classes_index).to(device).reshape(-1,1).long()

    mean_rank = 0.0
    n = len(valid_data)
    for i, (c, r, d) in enumerate(valid_data):
        c_data = torch.cat((c.repeat(classes_index.shape[0], 1), torch.Tensor([0]).repeat(classes_index.shape[0], 1).to(device).long(), classes_index), 1) 
        nf1_min = model.min_embedding[c_data[:,[0,2]]]
        nf1_delta = model.delta_embedding[c_data[:,[0,2]]]
        nf1_max = nf1_min+torch.exp(nf1_delta)
        boxes1 = Box(nf1_min[:, 0, :], nf1_max[:, 0, :])
        boxes2 = Box(nf1_min[:, 1, :], nf1_max[:, 1, :])
        c_probs = 1- compute_cond_probs(model, boxes1, boxes2).cpu().detach().numpy()
        index = rankdata(c_probs, method='average')
        dx = list(classes_index[:,0]).index(d)
        rank = index[dx]
        mean_rank += rank 
    mean_rank /= n
    return mean_rank

def compute_rank(model, valid_data, ratio, classes,device):
    classes_index = list(classes.values())
    classes_index = torch.Tensor(classes_index).to(device).reshape(-1,1).long()

    rank_values = []
    top1 = 0
    top10 = 0
    top100 = 0
    n = len(valid_data)
    rank_percentile = []
    for i, (c, r, d) in enumerate(valid_data):
        c_data = torch.cat((c.repeat(classes_index.shape[0], 1), torch.Tensor([0]).repeat(classes_index.shape[0], 1).to(device).long(), classes_index), 1) 
        nf1_min = model.min_embedding[c_data[:,[0,2]]]
        nf1_delta = model.delta_embedding[c_data[:,[0,2]]]
        nf1_max = nf1_min+torch.exp(nf1_delta)
        boxes1 = Box(nf1_min[:, 0, :], nf1_max[:, 0, :])
        boxes2 = Box(nf1_min[:, 1, :], nf1_max[:, 1, :])
        c_probs = 1- compute_cond_probs(model, boxes1, boxes2).cpu().detach().numpy()
        index = rankdata(c_probs, method='average')
        dx = list(classes_index[:,0]).index(d)
        rank = index[dx]
        rank_values.append(rank)
        rank_percentile.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 10:
            top10 += 1
        if rank <= 100:
            top100 += 1
    
    top1 /= (i+1)
    top10 /= (i+1)
    top100 /= (i+1)
    mean_rank = np.mean(rank_values)
    median_rank = statistics.median(rank_values)
    rank_percentile.sort()
    per_rank = np.percentile(rank_percentile,ratio)
    rank_dicts = dict(Counter(rank_values))
    nb_classes = model.min_embedding.shape[0]
    auc = compute_rank_roc(rank_dicts,nb_classes)
    return top1, top10, top100, mean_rank, median_rank, per_rank, auc, rank_values

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








