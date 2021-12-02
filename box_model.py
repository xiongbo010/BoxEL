#!/usr/bin/env python
# coding: utf-8

# In[59]:



import time
import torch
import math
import torch.nn as nn
from basic_box import Box
import torch.nn.functional as F
from torch.distributions import uniform


eps = 1e-8

def l2_side_regularizer(box, log_scale: bool = True):
    """Applies l2 regularization on all sides of all boxes and returns the sum.
    """
    min_x = box.min_embed 
    delta_x = box.delta_embed  

    if not log_scale:
        return torch.mean(delta_x ** 2)
    else:
        return torch.mean(F.relu(min_x + delta_x - 1 + eps )) +  torch.mean(F.relu(-min_x - eps)) #+ F.relu(torch.norm(min_x, p=2)-1)

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
        
        nf1_neg_min = self.min_embedding[data[5][:,[0,2]]]
        nf1_neg_delta = self.delta_embedding[data[5][:,[0,2]]]
        nf1_neg_max = nf1_neg_min+torch.exp(nf1_neg_delta)
        boxes1 = Box(nf1_neg_min[:, 0, :], nf1_neg_max[:, 0, :])
        boxes2 = Box(nf1_neg_min[:, 1, :], nf1_neg_max[:, 1, :])
        nf1_neg_loss, nf1_neg_reg_loss = self.nf1_loss(boxes1, boxes2)
        nf1_neg_loss = 1 - nf1_neg_loss


        # role inclusion
        translation_1 = self.relation_embedding[data[6][:,0]]
        translation_2 = self.relation_embedding[data[6][:,1]]
        scaling_1 = self.scaling_embedding[data[6][:,0]]
        scaling_2 = self.scaling_embedding[data[6][:,1]]
        role_inclusion_loss = self.role_inclusion_loss(translation_1,translation_2,scaling_1,scaling_2)
        # role chain
        translation_1 = self.relation_embedding[data[7][:,0]]
        translation_2 = self.relation_embedding[data[7][:,1]]
        translation_3 = self.relation_embedding[data[7][:,2]]
        scaling_1 = self.scaling_embedding[data[7][:,0]]
        scaling_2 = self.scaling_embedding[data[7][:,1]]
        scaling_3 = self.scaling_embedding[data[7][:,2]]
        role_chain_loss = self.role_chain_loss(translation_1,translation_2,translation_3,scaling_1,scaling_2,scaling_3)
        
        return nf1_loss.sum(), nf1_neg_loss.sum(), nf2_loss.sum(), nf3_loss.sum(), nf4_loss.sum(), disjoint_loss.sum(), role_inclusion_loss.sum(),role_chain_loss.sum(),nf1_reg_loss, nf1_neg_reg_loss, nf2_reg_loss , nf3_reg_loss , nf4_reg_loss, disjoint_reg_loss

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
        return self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) 
        
    def disjoint_loss(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_boxes1 = torch.log(torch.clamp(self.volumes(boxes1), 1e-10, 1e4))
        log_boxes2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        union = log_boxes1 + log_boxes2
        return torch.exp(log_intersection-union), l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)
        
    def role_inclusion_loss(self, translation_1,translation_2,scaling_1,scaling_2):
        loss_1 = torch.norm(translation_1-translation_2, p=2, dim=1,keepdim=True)
        loss_2 = torch.norm(F.relu(scaling_1/(scaling_2 +eps) -1), p=2, dim=1,keepdim=True)
        return loss_1+loss_2
    
    def role_chain_loss(self, translation_1,translation_2,translation_3,scaling_1,scaling_2,scaling_3):
        loss_1 = torch.norm(scaling_1*translation_1 + translation_2 -translation_3, p=2, dim=1,keepdim=True)
        loss_2 = torch.norm(F.relu(scaling_1*scaling_2/(scaling_3 +eps) -1), p=2, dim=1,keepdim=True)
        return loss_1+loss_2
    
    def init_concept_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed


class BoxELPPI(nn.Module):
    
    def __init__(self, vocab_size, relation_size,embed_dim, min_init_value, delta_init_value, relation_init_value, scaling_init_value, args):
        super(BoxELPPI, self).__init__()
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
        
        nf3_min = self.min_embedding[data[5][:,[0,2]]]
        nf3_delta = self.delta_embedding[data[5][:,[0,2]]]
        nf3_max = nf3_min+torch.exp(nf3_delta)
        relation = self.relation_embedding[data[5][:,1]]
        scaling = self.scaling_embedding[data[5][:,1]]
        boxes1 = Box(nf3_min[:, 0, :], nf3_max[:, 0, :])
        boxes2 = Box(nf3_min[:, 1, :], nf3_max[:, 1, :])
        nf3_neg_loss,nf3_neg_reg_loss = self.nf3_neg_loss(boxes1, relation, scaling, boxes2)
        # role inclusion
        # translation_1 = self.relation_embedding[data[6][:,0]]
        # translation_2 = self.relation_embedding[data[6][:,1]]
        # scaling_1 = self.scaling_embedding[data[6][:,0]]
        # scaling_2 = self.scaling_embedding[data[6][:,1]]
        # role_inclusion_loss = self.role_inclusion_loss(translation_1,translation_2,scaling_1,scaling_2)
        # role chain
        # translation_1 = self.relation_embedding[data[7][:,0]]
        # translation_2 = self.relation_embedding[data[7][:,1]]
        # translation_3 = self.relation_embedding[data[7][:,2]]
        # scaling_1 = self.scaling_embedding[data[7][:,0]]
        # scaling_2 = self.scaling_embedding[data[7][:,1]]
        # scaling_3 = self.scaling_embedding[data[7][:,2]]
        # role_chain_loss = self.role_chain_loss(translation_1,translation_2,translation_3,scaling_1,scaling_2,scaling_3)
        
        return nf1_loss.sum(), nf2_loss.sum(), nf3_loss.sum(), nf4_loss.sum(), disjoint_loss.sum(), 0, 0 ,nf3_neg_loss.sum(), nf1_reg_loss, nf2_reg_loss , nf3_reg_loss , nf4_reg_loss, disjoint_reg_loss,nf3_neg_reg_loss

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
        return self.inclusion_loss(trans_boxes, boxes2), l2_side_regularizer(trans_boxes, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) 
        
    def disjoint_loss(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_boxes1 = torch.log(torch.clamp(self.volumes(boxes1), 1e-10, 1e4))
        log_boxes2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        union = log_boxes1 + log_boxes2
        return torch.exp(log_intersection-union), l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True)
        
    def role_inclusion_loss(self, translation_1,translation_2,scaling_1,scaling_2):
        loss_1 = torch.norm(translation_1-translation_2, p=2, dim=1,keepdim=True)
        loss_2 = torch.norm(F.relu(scaling_1/(scaling_2 +eps) -1), p=2, dim=1,keepdim=True)
        return loss_1+loss_2
    
    def role_chain_loss(self, translation_1,translation_2,translation_3,scaling_1,scaling_2,scaling_3):
        loss_1 = torch.norm(scaling_1*translation_1 + translation_2 -translation_3, p=2, dim=1,keepdim=True)
        loss_2 = torch.norm(F.relu(scaling_1*scaling_2/(scaling_3 +eps) -1), p=2, dim=1,keepdim=True)
        return loss_1+loss_2

    def nf3_neg_loss(self, boxes1, relation, scaling, boxes2):
        trans_min = boxes1.min_embed*(scaling + eps) + relation
        trans_max = boxes1.max_embed*(scaling + eps) + relation
        trans_boxes = Box(trans_min, trans_max)
        return 1-self.inclusion_loss(trans_boxes, boxes2),l2_side_regularizer(trans_boxes, log_scale=True) + l2_side_regularizer(boxes1, log_scale=True) + l2_side_regularizer(boxes2, log_scale=True) 
        
    
    def init_concept_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed



