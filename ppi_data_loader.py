#!/usr/bin/env python
# coding: utf-8



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
    rel = f'<http://interacts>'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data

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


def load_data(filename):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': [],'nf_inclusion':[],'nf_chain':[] }
    with open(filename) as f:
        for line in f:
            # Ignore SubObjectPropertyOf
            if line.startswith('SubObjectPropertyOf'):
                line = line.strip()[20:-1]
                if line.startswith('ObjectPropertyChain'):
                    line_chain = line.strip()[20:]
                    line1 = line_chain.split(")")
                    line2 = line1[0].split(' ')
                    r1 = line2[0].strip()
                    r2 = line2[1].strip()
                    r3 = line1[1].strip()
                    if r1.startswith('<http://') and r2.startswith('<http://') and r3.startswith('<http://'):
                        if r1 not in relations:
                            relations[r1] = len(relations)
                        if r2 not in relations:
                            relations[r2] = len(relations)
                        if r3 not in relations:
                            relations[r3] = len(relations)
                        data['nf_chain'].append((relations[r1],relations[r2],relations[r3]))
                else:
                    it = line.split(' ')
                    r1 = it[0]
                    r2 = it[1]
                    if r1.startswith('<http://') and r2.startswith('<http://'):
                        if r1 not in relations:
                            relations[r1] = len(relations)
                        if r2 not in relations:
                            relations[r2] = len(relations)
                        data['nf_inclusion'].append((relations[r1], relations[r2]))
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
                # if r =='<http://interacts>':
                #     data['interact'].append((classes[c], relations[r], classes[d]))
                # elif r =='<http://hasFunction>':
                #     data['hasFunction'].append((classes[c], relations[r], classes[d]))
                # else:
                data['nf3'].append((classes[c], relations[r], classes[d]))
            else:
                # C SubClassOf D
                it = line.split(' ')
                c = it[0]
                d = it[1]
                r = 'SubClassOf'
                # if r not in relations:
                #     relations[r] = len(relations)
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['nf1'].append((classes[c],-1,classes[d]))
                
    # Check if TOP in classes and insert if it is not there
    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
    if 'owl:Nothing' not in classes:
        classes['owl:Nothing'] = len(classes)

    prot_ids = []
    for k, v in classes.items():
        if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
            prot_ids.append(v)
    prot_ids = np.array(prot_ids)
    
    # Add at least one disjointness axiom if there is 0
    if len(data['disjoint']) == 0:
        nothing = classes['owl:Nothing']
        n_prots = len(prot_ids)
        for i in range(10):
            it = np.random.choice(n_prots, 2)
            if it[0] != it[1]:
                data['disjoint'].append(
                    (prot_ids[it[0]], prot_ids[it[1]], nothing))
                break
    # Add corrupted triples nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    inter_ind = 0
    for k, v in relations.items():
        if k == '<http://interacts>':
            inter_ind = v
 
    for c, r, d in data['nf3']:
        if r != inter_ind:
            continue
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
    data['nf_inclusion'] = np.array(data['nf_inclusion'])
    data['nf_chain'] = np.array(data['nf_chain'])
    data['nf3_neg'] = np.array(data['nf3_neg'])

    # data['interact'] = np.array(data['interact'])
    # data['interact_neg'] = np.array(data['interact_neg'])
    # data['hasFunction'] = np.array(data['hasFunction'])
                            
    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]
    
    return data, classes, relations
#original Loss

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
            nf_inclusion_index = np.random.choice(self.data['nf_inclusion'].shape[0],self.batch_size)
            nf_chain_index = np.random.choice(self.data['nf_chain'].shape[0],self.batch_size)
            # interact_index= np.random.choice(
            #     self.data['interact'].shape[0], self.batch_size)
            nf3_neg_index = np.random.choice(
                self.data['nf3_neg'].shape[0], self.batch_size)
            # hasFunction_index = np.random.choice(
            #     self.data['hasFunction'].shape[0], self.batch_size)
            nf1 = self.data['nf1'][nf1_index]
            nf2 = self.data['nf2'][nf2_index]
            nf3 = self.data['nf3'][nf3_index]
            nf4 = self.data['nf4'][nf4_index]
            dis = self.data['disjoint'][dis_index]
            top = self.data['top'][top_index]
            nf_inclusion = self.data['nf_inclusion'][nf_inclusion_index]
            nf_chain = self.data['nf_chain'][nf_chain_index]
            # interact = self.data['interact'][interact_index]
            nf3_neg = self.data['nf3_neg'][nf3_neg_index]
            # hasFunction = self.data['hasFunction'][hasFunction_index]
            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1
            return ([nf1, nf2, nf3, nf4, dis,nf3_neg, nf_inclusion,nf_chain,top], labels)
        else:
            self.reset()