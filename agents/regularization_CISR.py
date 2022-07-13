#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 7:50 PM
# @Author  : Jingyang.Zhang
'''
Selective regularization with comprehensive importance (shape and semantic)
'''
import torch
from .default_CISR import SegNormalNN_CISR
import torch.nn as nn
import numpy as np
from utils.dropout_utils import apply_dropout, close_dropout

class L2_CISR(SegNormalNN_CISR):
    '''
    apply regularization for all parameters in SegNormalNN_CISR
    '''
    def __init__(self, agent_config):
        super(L2_CISR, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.regularization_terms = {}
        self.task_count = 0
        self.online_reg = True

    def calculate_importance(self, dataloader):
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)
        return importance

    def learn_batch(self, site_name, train_loader, val_loader=None):
        self.log('#reg_term:', len(self.regularization_terms))

        super(L2_CISR, self).learn_batch(site_name, train_loader, val_loader)

        print('Save current model parameters...')
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        print('Compute importance matrix...')
        importance = self.calculate_importance(train_loader)

        self.task_count += 1
        if self.online_reg and len(self.regularization_terms) > 0:
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}

    def criterion(self, out_segs, out_tanhs, targets, regularization=True):
        loss_seg, loss_regression, loss_embedding, loss = super(L2_CISR, self).criterion(out_segs, out_tanhs, targets)

        if regularization and len(self.regularization_terms) > 0:
            loss_reg = 0

            for i, reg_term in self.regularization_terms.items():
                task_loss_reg = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_loss_reg += (importance[n] * (p - task_param[n]) ** 2).sum()
                loss_reg += task_loss_reg

            loss += self.config['reg_coef'] * loss_reg
        return loss_seg, loss_regression, loss_embedding, loss


class Reg_CISR(L2_CISR):
    def __init__(self, agent_config):
        super(Reg_CISR, self).__init__(agent_config)
        self.online_reg = True
        self.alpha = agent_config['alpha']
        self.alpha1 = agent_config['alpha1']

    def calculate_importance(self, dataloader):
        self.log('Computing comprehensive Importance.')

        if self.online_reg and len(self.regularization_terms)>0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)

        self.model.eval()

        for batch in dataloader:
            input, target = batch['img'], batch['gt']
            if self.gpu:
                input, target = batch['img'].cuda(), batch['gt'].cuda()
            out_seg, out_regression = self.forward(input)

            self.embedding.eval()
            seg_code, _ = self.embedding(out_seg)

            self.model = apply_dropout(self.model)
            with torch.no_grad():
                T = 8
                mc_seg = torch.zeros([T, out_seg.shape[0], out_seg.shape[1], out_seg.shape[2], out_seg.shape[3]])
                for mc_pass in range(T):
                    mc_seg_pass, _ = self.model.forward(input)
                    mc_seg[mc_pass,:,:,:,:] = mc_seg_pass
                mc_seg_mean = torch.mean(mc_seg, dim=0)
                mc_seg_uc = -1 * torch.sum(mc_seg_mean * torch.log(mc_seg_mean + 1e-6), dim=1, keepdim=True)
                mask = torch.exp(-1 * mc_seg_uc)
                # mask normalization
                for mask_i in range(mask.shape[0]):
                    mask_min, mask_max = mask[mask_i,:,:,:].min(), mask[mask_i,:,:,:].max()
                    mask[mask_i,:,:,:] = (mask[mask_i,:,:,:] - mask_min) / (mask_max - mask_min)
                    assert mask[mask_i,:,:,:].min() == 0 and mask[mask_i,:,:,:].max() == 1

            self.model = close_dropout(self.model)

            L2_pred = torch.mean(torch.pow(mask.cuda() * out_seg, 2)) + self.alpha * torch.mean(torch.pow(out_regression, 2)) + \
                self.alpha1 * torch.mean(torch.pow(seg_code, 2))
            self.model.zero_grad()
            L2_pred.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:
                    p += (self.params[n].grad.abs() / len(dataloader))

        self.model.train()
        return importance


