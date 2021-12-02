#!/usr/bin/env python
# coding: utf-8

# In[1]:



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


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
# model parameters
parser.add_argument('--model', type=str, default='softbox', help='model type: choose from softbox, gumbel')
parser.add_argument('--box_embedding_dim', type=int, default=40, help='box embedding dimension')
parser.add_argument('--softplus_temp', type=float, default=1.0, help='beta of softplus function')
# gumbel box parameters
parser.add_argument('--gumbel_beta', type=float, default=1.0, help='beta value for gumbel distribution')
parser.add_argument('--scale', type=float, default=1.0, help='scale value for gumbel distribution')

parser.add_argument('--dataset', type=str, default='GO', help='dataset')
parser.add_argument('--using_rbox', type=int, default=1, help='using_rbox')
parser.add_argument('--gpu', type=int, default=3, help='gpu')

parser.add_argument('--dimension', type=int, default=50, help='dimension')
parser.add_argument('--learning_rate', type=int, default=0.001, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--seed', type=int, default=1111, help='seed')

args = parser.parse_args(args=[])
args.save_to = "./checkpoints/" + args.model

gpu = args.gpu
dimension = args.dimension
learning_rate = args.learning_rate
batch_size = args.batch_size
seed = args.seed
dataset = args.dataset
using_rbox = args.using_rbox

torch.cuda.set_device(gpu)
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)
import random
random.seed(seed)


# In[3]:


from box_model import BoxELPPI as Model
import torch
from ppi_data_loader import load_cls,load_valid_data
from ppi_data_loader import load_data
from ppi_data_loader import Generator
from evaluation import compute_mean_rank, compute_rank, compute_accuracy


# In[4]:



dataset = 'PPI'

total_sub_cls = []
train_file = f"data/{dataset}/{dataset}_train.txt"
va_file = f"data/{dataset}/{dataset}_valid.txt"
test_file = f"data/{dataset}/{dataset}_test.txt"
train_sub_cls, train_samples = load_cls(train_file)
valid_sub_cls, valid_samples = load_cls(va_file)
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

nb_classes = len(classes)
nb_relations = len(relations)

print("no. classes:",nb_classes)
print("no. relations:",nb_relations)

for k in train_data:
    print(k, train_data[k].shape[0])


# In[5]:


train_data


# In[6]:



batch_size = 256 

org = 4932
proteins = {}
for k, v in classes.items():
    if k.startswith(f'<http://{org}'):
        proteins[k] = v
print('Proteins:', len(proteins))

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
classes_index.shape

protein_index = list(proteins.values())
protein_index = torch.Tensor(protein_index).to(device).reshape(-1,1).long()


# In[7]:


wandb.init(project="ppi", reinit=True, config=args)


# In[ ]:


torch.manual_seed(seed)
import random
random.seed(seed)
print(dataset, using_rbox, dimension,learning_rate,batch_size,gpu, seed)

model = Model(nb_classes, nb_relations, dimension, [1e-4, 0.2], [-0.1, 0], [-0.1,0.1], [0.9,1.1], args).to(device)
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
    nf1_neg_total_loss = 0.0
    role_chain_total_loss = 0.0
    role_inclusion_total_loss = 0.0
#     interact_total_loss = 0.0
    interact_neg_total_loss = 0.0
#     membership_total_loss = 0.0

    for step, batch in enumerate(train_generator):
        if step < steps_per_epoch:
            nf1_loss, nf2_loss, nf3_loss, nf4_loss, disjoint_loss, role_inclusion_loss, role_chain_loss, interact_neg_loss, nf1_reg_loss, nf2_reg_loss , nf3_reg_loss , nf4_reg_loss , disjoint_reg_loss, inter_neg_reg_loss = model(batch[0])
            loss =  nf1_loss + nf1_reg_loss + nf2_loss + nf2_reg_loss + nf3_loss + nf3_reg_loss + nf4_loss + disjoint_loss + disjoint_reg_loss + nf4_reg_loss +  interact_neg_loss + inter_neg_reg_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            nf1_total_loss += nf1_loss
            nf2_total_loss += nf2_loss
            nf3_total_loss += nf3_loss
            nf4_total_loss += nf4_loss
            disjoint_total_loss += disjoint_loss
            role_inclusion_total_loss += role_inclusion_loss
            role_chain_total_loss += role_inclusion_loss
#             interact_total_loss += interact_loss
            interact_neg_total_loss += interact_neg_loss
#             membership_total_loss += membership_loss
        else:
            train_generator.reset()
            break
    mean_rank = compute_mean_rank(model, valid_data[0:100],classes,device)
    valid_accuracy = compute_accuracy(model, valid_data)

    wandb.log({'train loss': train_loss.item()/(step+1),'nf1_loss':nf1_total_loss.item()/(step+1), 
               'nf2_loss':nf2_total_loss.item()/(step+1), 'nf3_loss':nf3_total_loss.item()/(step+1), 
               'nf4_loss':nf4_total_loss.item()/(step+1), 'disjoint_loss':disjoint_total_loss.item()/(step+1),
               'role_inclusion_loss':role_inclusion_total_loss.item()/(step+1),
               'role_chain_loss':role_chain_total_loss.item()/(step+1),
               'interact_total_loss':interact_total_loss.item()/(step+1),
               'interact_neg_total_loss':interact_neg_total_loss.item()/(step+1),
#                'membership_total_loss':membership_total_loss.item()/(step+1),
                'valid_accuracy':valid_accuracy }) 
    
    PATH = './models/box_el' + '_' + str(using_rbox)  + '_dim' + str(dimension) + "_lr" + str(learning_rate) + '_batch'+str(batch_size)+'_seed'+ str(seed)+ '.pt'
    if epoch % 10 == 0:
        torch.save(model, PATH)
        print('Epoch:%d' %(epoch + 1), "Train loss: %f" %(train_loss.item()/(step+1)), f'Valid Mean Rank: {mean_rank}\n')


top1,top10, top100, mean_rank, median_rank, per_rank, auc,rank_values = compute_rank(model, test_data, 90,classes,device)
acc = compute_accuracy(model, test_data)
print(top1,top10, top100, mean_rank, median_rank, per_rank, auc,acc)

