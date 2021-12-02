#!/usr/bin/env python
# coding: utf-8


import time
import wandb
import torch
from torch.utils.data import DataLoader
import math
import argparse
import os
import json

import torch
import wandb
import torch.nn as nn
from basic_box import Box
import torch.nn.functional as F
from torch.distributions import uniform

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training (eg. no nvidia GPU)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
# model parameters
parser.add_argument('--model', type=str, default='softbox', help='model type: choose from softbox, gumbel')
parser.add_argument('--box_embedding_dim', type=int, default=40, help='box embedding dimension')
parser.add_argument('--softplus_temp', type=float, default=1.0, help='beta of softplus function')
# gumbel box parameters
parser.add_argument('--gumbel_beta', type=float, default=1.0, help='beta value for gumbel distribution')
parser.add_argument('--scale', type=float, default=1.0, help='scale value for gumbel distribution')

parser.add_argument('--dimension', type=int, default=50, help='number of epochs to train')
parser.add_argument('--learning_rate', type=int, default=0.001, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256, help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1111, help='number of epochs to train')

args = parser.parse_args()
args.save_to = "./checkpoints/" + args.model

dimension = args.dimension
learning_rate = args.learning_rate
batch_size = args.batch_size
seed = args.seed

# for seed in [1111,2222,3333,4444,5555]:

torch.manual_seed(seed)
import random
random.seed(seed)

eps = 1e-8

def l2_side_regularizer(box, log_scale: bool = True):
    """Applies l2 regularization on all sides of all boxes and returns the sum.
    """
    min_x = box.min_embed 
    delta_x = box.delta_embed  

    if not log_scale:
        return torch.mean(delta_x ** 2)
    else:
        return  torch.mean(F.relu(min_x + delta_x - 1 + eps )) +F.relu(torch.norm(min_x, p=2)-1)



euler_gamma = 0.57721566490153286060

gamma = 1

class BoxEL(nn.Module):
    
    def __init__(self, vocab_size, relation_size,embed_dim, min_init_value, delta_init_value, relation_init_value, scaling_init_value, args):
        super(BoxEL, self).__init__()
        min_embedding = self.init_concept_embedding(vocab_size, embed_dim, min_init_value)
        delta_embedding = self.init_concept_embedding(vocab_size, embed_dim, delta_init_value)
        relation_embedding = self.init_concept_embedding(relation_size, embed_dim, relation_init_value)
        scaling_embedding = self.init_concept_embedding(relation_size, embed_dim, scaling_init_value)
        
        self.temperature = args.softplus_temp
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)
        self.relation_embedding = nn.Parameter(relation_embedding)
        self.scaling_embedding = nn.Parameter(scaling_embedding)
        
        self.gumbel_beta = args.gumbel_beta
        self.scale = args.scale

    def forward(self, data):
        nf1_min = self.min_embedding[data[0][:,[0,2]]]
        nf1_delta = self.delta_embedding[data[0][:,[0,2]]]
        nf1_max = nf1_min+torch.exp(nf1_delta)
        
        boxes1 = Box(nf1_min[:, 0, :], nf1_max[:, 0, :])
        boxes2 = Box(nf1_min[:, 1, :], nf1_max[:, 1, :])
        
        nf1_loss, nf1_reg_loss = self.nf1_loss(boxes1, boxes2)
        
        nf2_min = self.min_embedding[data[1]]
        nf2_delta = self.delta_embedding[data[1]]
        nf2_max = nf2_min+torch.exp(nf2_delta)
        
        boxes1 = Box(nf2_min[:, 0, :], nf2_max[:, 0, :])
        boxes2 = Box(nf2_min[:, 1, :], nf2_max[:, 1, :])
        boxes3 = Box(nf2_min[:, 2, :], nf2_max[:, 2, :])
        
        nf2_loss,nf2_reg_loss = self.nf2_loss(boxes1, boxes2, boxes3)
        
        nf3_min = self.min_embedding[data[2][:,[0,2]]]
        nf3_delta = self.delta_embedding[data[2][:,[0,2]]]
        nf3_max = nf3_min+torch.exp(nf3_delta)
        relation = self.relation_embedding[data[2][:,1]]
        scaling = self.scaling_embedding[data[2][:,1]]
        
        boxes1 = Box(nf3_min[:, 0, :], nf3_max[:, 0, :])
        boxes2 = Box(nf3_min[:, 1, :], nf3_max[:, 1, :])
        
        nf3_loss,nf3_reg_loss = self.nf3_loss(boxes1, relation, scaling, boxes2)
        
        nf4_min = self.min_embedding[data[3][:,1:]]
        nf4_delta = self.delta_embedding[data[3][:,1:]]
        nf4_max = nf4_min+torch.exp(nf4_delta)
        relation = self.relation_embedding[data[3][:,0]]
        scaling = self.scaling_embedding[data[3][:,0]]
        
        boxes1 = Box(nf4_min[:, 0, :], nf4_max[:, 0, :])
        boxes2 = Box(nf4_min[:, 1, :], nf4_max[:, 1, :])
        
        nf4_loss,nf4_reg_loss = self.nf4_loss(relation, scaling, boxes1, boxes2)
        
        disjoint_min = self.min_embedding[data[4]]
        disjoint_delta = self.delta_embedding[data[4]]
        disjoint_max = disjoint_min+torch.exp(disjoint_delta)
        boxes1 = Box(disjoint_min[:, 0, :], disjoint_max[:, 0, :])
        boxes2 = Box(disjoint_min[:, 1, :], disjoint_max[:, 1, :])
        disjoint_loss,disjoint_reg_loss = self.disjoint_loss(boxes1, boxes2)
        
        nf3_neg_min = self.min_embedding[data[6][:,[0,2]]]
        nf3_neg_delta = self.delta_embedding[data[6][:,[0,2]]]
        nf3_neg_max = nf3_neg_min+torch.exp(nf3_neg_delta)
        
        relation = self.relation_embedding[data[6][:,1]]
        scaling = self.scaling_embedding[data[6][:,1]]
        
        boxes1 = Box(nf3_neg_min[:, 0, :], nf3_neg_max[:, 0, :])
        boxes2 = Box(nf3_neg_min[:, 1, :], nf3_neg_max[:, 1, :])
        
        nf3_neg_loss,nf3_neg_reg_loss = self.nf3_neg_loss(boxes1, relation, scaling, boxes2)
        
        all_min = self.min_embedding
        all_delta = self.delta_embedding
        all_max = all_min+torch.exp(all_delta)
        boxes = Box(all_min, all_max)
        reg_loss = l2_side_regularizer(boxes, log_scale=True)
        
        return nf1_loss.sum(), nf2_loss.sum(), nf3_loss.sum(), nf4_loss.sum(), disjoint_loss.sum(), nf3_neg_loss.sum(), nf1_reg_loss, nf2_reg_loss , nf3_reg_loss , nf4_reg_loss , disjoint_reg_loss , nf3_neg_reg_loss

    def get_cond_probs(self, data):
        nf3_min = self.min_embedding[data[:,[0,2]]]
        nf3_delta = self.delta_embedding[data[:,[0,2]]]
        nf3_max = nf3_min+torch.exp(nf3_delta)
        
        relation = self.relation_embedding[data[:,1]]
        
        boxes1 = Box(nf3_min[:, 0, :], nf3_max[:, 0, :])
        boxes2 = Box(nf3_min[:, 1, :], nf3_max[:, 1, :])
        
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_box2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        return torch.exp(log_intersection-log_box2)
        

    def volumes(self, boxes):
        return F.softplus(boxes.delta_embed, beta=self.temperature).prod(1)

    def intersection(self, boxes1, boxes2):
        intersections_min = torch.max(boxes1.min_embed, boxes2.min_embed)
        intersections_max = torch.min(boxes1.max_embed, boxes2.max_embed)
        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box
    
    def inclusion_loss(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_box1 = torch.log(torch.clamp(self.volumes(boxes1), 1e-10, 1e4))
        
        return 1-torch.exp(log_intersection-log_box1)
    
    def nf1_loss(self, boxes1, boxes2):
        return self.inclusion_loss(boxes1, boxes2), l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)
        
    def nf2_loss(self, boxes1, boxes2, boxes3):
        inter_box = self.intersection(boxes1, boxes2)
        return self.inclusion_loss(inter_box, boxes3), l2_side_regularizer(inter_box, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) + l2_side_regularizer(boxes3, log_scale=True)
    
    def nf3_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed*(scaling + eps) + relation
        trans_max = boxes1.max_embed*(scaling + eps) + relation
        trans_boxes = Box(trans_min, trans_max)
        return self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) 
    
    def nf4_loss(self, relation, scaling, boxes1, boxes2):
        trans_min = (boxes1.min_embed - relation)/(scaling + eps)
        trans_max = (boxes1.max_embed - relation)/(scaling + eps)
        trans_boxes = Box(trans_min, trans_max)
#         log_trans_boxes = torch.log(torch.clamp(self.volumes(trans_boxes), 1e-10, 1e4))
        return self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) 
        
    def disjoint_loss(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_boxes1 = torch.log(torch.clamp(self.volumes(boxes1), 1e-10, 1e4))
        log_boxes2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        union = log_boxes1 + log_boxes2
        return torch.exp(log_intersection-union), l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)
        
    def nf3_neg_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed*(scaling + eps) + relation
        trans_max = boxes1.max_embed*(scaling + eps) + relation
        trans_boxes = Box(trans_min, trans_max)
#         trans_min = boxes1.min_embed + relation
#         trans_max = trans_min + torch.clamp((boxes1.max_embed - boxes1.min_embed)*(scaling + eps), 1e-10, 1e4)
#         trans_boxes = Box(trans_min, trans_max)
        return 1-self.inclusion_loss(trans_boxes, boxes2),l2_side_regularizer(trans_boxes, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) 
        
    def init_concept_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed


# In[75]:




import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import click as ck
import numpy as np
import pandas as pd

import re
import math
import matplotlib.pyplot as plt
import logging
from scipy.stats import rankdata

def load_valid_data(valid_data_file, classes, relations):
    data = []
    rel = f'SubClassOf'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = it[0]
            id2 = it[1]
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data


# In[4]:


def load_cls(train_data_file):
    train_subs=list()
    counter=0
    with open(train_data_file,'r') as f:
        for line in f:
            counter+=1
            it = line.strip().split()
            cls1 = it[0]
            cls2 = it[1]
            train_subs.append(cls1)
            train_subs.append(cls2)
    train_cls = list(set(train_subs))
    return train_cls,counter


#Original Loss
def load_data(filename):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': []}
    with open(filename) as f:
        for line in f:
            # Ignore SubObjectPropertyOf
            if line.startswith('SubObjectPropertyOf'):
                continue
            # Ignore SubClassOf()
            line = line.strip()[11:-1]
            if not line:
                continue
            if line.startswith('ObjectIntersectionOf('):
                # C and D SubClassOf E
                it = line.split(' ')
                c = it[0][21:]
                d = it[1][:-1]
                e = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if e not in classes:
                    classes[e] = len(classes)
                form = 'nf2'
                if e == 'owl:Nothing':
                    form = 'disjoint'
                data[form].append((classes[c], classes[d], classes[e]))
                
            elif line.startswith('ObjectSomeValuesFrom('):
                # R some C SubClassOf D
                it = line.split(' ')
                r = it[0][21:]
                c = it[1][:-1]
                d = it[2]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf4'].append((relations[r], classes[c], classes[d]))
            elif line.find('ObjectSomeValuesFrom') != -1:
                # C SubClassOf R some D
                it = line.split(' ')
                c = it[0]
                r = it[1][21:]
                d = it[2][:-1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                if r not in relations:
                    relations[r] = len(relations)
                data['nf3'].append((classes[c], relations[r], classes[d]))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]
                r = 'SubClassOf'
                if r not in relations:
                    relations[r] = len(relations)
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf1'].append((classes[c],relations[r],classes[d]))
                
    # Check if TOP in classes and insert if it is not there
    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
#changing by adding sub classes of train_data ids to prot_ids
    prot_ids = []
    class_keys = list(classes.keys())
    for val in all_subcls:
        if val not in class_keys:
            cid = len(classes)
            classes[val] = cid
            prot_ids.append(cid)
        else:
            prot_ids.append(classes[val])

    prot_ids = np.array(prot_ids)
    
    
    # Add corrupted triples nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    for c, r, d in data['nf3']:
        x = np.random.choice(prot_ids)
        while x == c:
            x = np.random.choice(prot_ids)
            
        y = np.random.choice(prot_ids)
        while y == d:
             y = np.random.choice(prot_ids)
        data['nf3_neg'].append((c, r,x))
        data['nf3_neg'].append((y, r, d))
        
    
    data['nf1'] = np.array(data['nf1'])
    data['nf2'] = np.array(data['nf2'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['disjoint'] = np.array(data['disjoint'])
    data['top'] = np.array([classes['owl:Thing'],])
    data['nf3_neg'] = np.array(data['nf3_neg'])
                            
    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]
    
    return data, classes, relations


class Generator(object):
    def __init__(self, data, batch_size=128, steps=100):
        self.data = data
        self.batch_size = batch_size
        self.steps = steps
        self.start = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.steps:
            nf1_index = np.random.choice(
                self.data['nf1'].shape[0], self.batch_size)
            nf2_index = np.random.choice(
                self.data['nf2'].shape[0], self.batch_size)
            nf3_index = np.random.choice(
                self.data['nf3'].shape[0], self.batch_size)
            nf4_index = np.random.choice(
                self.data['nf4'].shape[0], self.batch_size)
            dis_index = np.random.choice(
                self.data['disjoint'].shape[0], self.batch_size)
            top_index = np.random.choice(
                self.data['top'].shape[0], self.batch_size)
            nf3_neg_index = np.random.choice(
                self.data['nf3_neg'].shape[0], self.batch_size)
            nf1 = self.data['nf1'][nf1_index]
            nf2 = self.data['nf2'][nf2_index]
            nf3 = self.data['nf3'][nf3_index]
            nf4 = self.data['nf4'][nf4_index]
            dis = self.data['disjoint'][dis_index]
            top = self.data['top'][top_index]
            nf3_neg = self.data['nf3_neg'][nf3_neg_index]
            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            return ([nf1, nf2, nf3, nf4, dis, top, nf3_neg], labels)
        else:
            self.reset()



from scipy.stats import rankdata
import statistics
from collections import Counter

def compute_cond_probs(model, boxes1, boxes2):
    log_intersection = torch.log(torch.clamp(model.volumes(model.intersection(boxes1, boxes2)), 1e-10, 1e4))
    log_box2 = torch.log(torch.clamp(model.volumes(boxes2), 1e-10, 1e4))
    return torch.exp(log_intersection-log_box2)

def compute_mean_rank(model, valid_data):
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
        rank = index[d]
        mean_rank += rank 
    mean_rank /= n
    return mean_rank

def compute_rank(model, valid_data, ratio):
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
        rank = index[d]
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


dataset = 'GO'

total_sub_cls=[]
train_file = f"data/{dataset}/{dataset}_train.txt"
va_file = f"data/{dataset}/{dataset}_valid.txt"
test_file = f"data/{dataset}/{dataset}_test.txt"
train_sub_cls,train_samples = load_cls(train_file)
valid_sub_cls,valid_samples = load_cls(va_file)
test_sub_cls,test_samples = load_cls(test_file)
total_sub_cls = train_sub_cls + valid_sub_cls + test_sub_cls
all_subcls = list(set(total_sub_cls))

print("Training data samples:",train_samples)
print("Training data classes:",len(train_sub_cls))

gdata_file=f"data/{dataset}/{dataset}_latest_norm_mod.owl"
train_data, classes, relations = load_data(gdata_file)
valid_data_file=f"data/{dataset}/{dataset}_valid.txt"
valid_data = load_valid_data(valid_data_file, classes, relations)
valid_data = torch.Tensor(valid_data).long().to(device)


proteins = {} # substitute for classes with subclass case
for val in all_subcls:
    proteins[val] = classes[val]
nb_classes = len(classes)
nb_relations = len(relations)

print("no. classes:",nb_classes)
print("no. relations:",nb_relations)
nb_train_data = 0

for key, val in train_data.items():
    nb_train_data = max(len(val), nb_train_data)
train_steps = int(math.ceil(nb_train_data / (1.0 * batch_size)))
train_generator = Generator(train_data, batch_size, steps=train_steps)

cls_dict = {v: k for k, v in classes.items()}
rel_dict = {v: k for k, v in relations.items()}

cls_list = []
rel_list = []
for i in range(nb_classes):
    cls_list.append(cls_dict[i])
for i in range(nb_relations):
    rel_list.append(rel_dict[i])

classes_index = list(classes.values())
classes_index = torch.Tensor(classes_index).to(device).reshape(-1,1).long()



for seed in [1111,2222,3333,4444,5555,6666,7777,8888,9999,1010]:

    torch.manual_seed(seed)
    import random
    random.seed(seed)
    print(seed)

    wandb.init(project="basic_box", reinit=True, config=args)

    model = BoxEL(nb_classes, nb_relations, dimension, [1e-4, 0.2], [-0.1, 0], [-0.1,0.1], [0.9,1.1], args).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    wandb.watch(model)

    model.train()
    steps_per_epoch = train_steps
    train_generator.reset()

    for epoch in range(args.epochs):
        train_loss = 0.0
        nf1_total_loss = 0.0
        nf2_total_loss = 0.0
        nf3_total_loss = 0.0
        nf4_total_loss = 0.0
        disjoint_total_loss = 0.0
        nf3_neg_total_loss = 0.0
        
        for step, batch in enumerate(train_generator):
            if step < steps_per_epoch:
                nf1_loss, nf2_loss, nf3_loss, nf4_loss, disjoint_loss, nf3_neg_loss, nf1_reg_loss, nf2_reg_loss , nf3_reg_loss , nf4_reg_loss , disjoint_reg_loss , nf3_neg_reg_loss = model(batch[0])
                loss =  nf1_loss + nf1_reg_loss + nf2_loss + nf2_reg_loss + disjoint_loss + disjoint_reg_loss + nf3_loss + nf3_reg_loss + nf4_loss + nf4_reg_loss + nf3_neg_loss + nf3_neg_reg_loss
                assert torch.isnan(loss).sum() == 0
                optimizer.zero_grad()
                loss.backward()
                assert torch.isnan(model.min_embedding).sum() == 0
                optimizer.step()
                assert torch.isnan(model.min_embedding).sum() == 0
                assert torch.isnan(model.min_embedding.grad).sum() == 0
                train_loss += loss
                nf1_total_loss += nf1_loss
                nf2_total_loss += nf2_loss
                nf3_total_loss += nf3_loss
                nf4_total_loss += nf4_loss
                nf3_neg_total_loss += nf3_neg_loss
            else:
                train_generator.reset()
                break
                
        mean_rank = compute_mean_rank(model, valid_data[0:100])
        valid_accuracy = compute_accuracy(model, valid_data)
        
        wandb.log({'train loss': train_loss.item()/(step+1),'nf1_loss':nf1_total_loss.item()/(step+1), 'nf2_loss':nf2_total_loss.item()/(step+1), 'nf3_loss':nf3_total_loss.item()/(step+1), 'nf4_loss':nf4_total_loss.item()/(step+1), 'nf3_neg_loss':nf3_neg_total_loss.item()/(step+1), 'mean_rank':mean_rank,'valid_accuracy':valid_accuracy })

        PATH = './models/box_el_go_dim' + str(dimension) + "_lr" + str(learning_rate) + '_batch'+str(batch_size)+'_seed'+ str(seed)+ '.pt'
        if epoch % 20 == 0:
            torch.save(model, PATH)
            print('Epoch:%d' %(epoch + 1), "Train loss: %f" %(train_loss.item()/(step+1)), f'Valid Mean Rank: {mean_rank}\n')

    test_file = f'data/{dataset}/{dataset}_test.txt'
    test_data = load_valid_data(test_file,classes,relations)
    test_data = torch.Tensor(test_data).long().to(device)

    top1,top10, top100, mean_rank, median_rank, per_rank, auc,rank_values = compute_rank(model, test_data, 90)
    print(seed, top1,top10, top100, mean_rank, median_rank, per_rank, auc, compute_accuracy(model, test_data))

    test_file = f'data/{dataset}/{dataset}_inferences.txt'
    test_data = load_valid_data(test_file,classes,relations)
    test_data = torch.Tensor(test_data).long().to(device)
    top1,top10, top100, mean_rank, median_rank, per_rank, auc,rank_values = compute_rank(model, test_data, 90)
    print(seed, top1,top10, top100, mean_rank, median_rank, per_rank, auc, compute_accuracy(model, test_data))








