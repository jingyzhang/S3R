#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/23 9:23 PM
# @Author  : Jingyang.Zhang
'''
training model with regression head and embedding head
'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from utils.compute_sdf import compute_sdf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .default import SegNormalNN, dice_loss
from sklearn.metrics import f1_score
import numpy as np



class SegNormalNN_CISR(SegNormalNN):
    def __init__(self, agent_config):
        super(SegNormalNN_CISR, self).__init__(agent_config)
        self.beta = agent_config['beta']
        self.beta1 = agent_config['beta1']

        self.embedding = models.__dict__['embedding'].__dict__['embedding']()
        embedding_path = '/opt/data/private/zhangjingyang/pretrain/embedding.pth'
        print('==> Load model weights:', embedding_path)
        embedding_state = torch.load(embedding_path, map_location=lambda storage, loc:storage)
        self.embedding.load_state_dict(embedding_state)
        self.embedding.cuda()
        for name, param in self.embedding.named_parameters():
            param.requires_grad = False

    def learn_batch(self, site_name, train_loader, val_loader=None):
        if self.config['reset_optimizer']:
            self.log('Optimizer is reset!')
            self.init_optimizer()
        else:
            self.optimizer.param_groups[0]['lr'] = self.config['lr']

        for self.epoch in range(self.config['epoches']):
            losses_seg = AverageMeter()
            losses_regression = AverageMeter()
            losses_embedding = AverageMeter()
            losses = AverageMeter()

            self.model.train()
            self.scheduler.step()

            with tqdm(total=len(train_loader), desc='Epoch %d/%d' % (self.epoch+1,self.config['epoches']), unit='batch') \
                    as pbar:
                for batch in train_loader:
                    imgs, gts = batch['img'], batch['gt']
                    if self.gpu:
                        imgs, gts = imgs.cuda(), gts.cuda()

                    loss_seg, loss_regression, loss_embedding, loss = self.update_model(imgs, gts)

                    losses_seg.update(loss_seg, imgs.shape[0])
                    losses_regression.update(loss_regression, imgs.shape[0])
                    losses_embedding.update(loss_embedding, imgs.shape[0])
                    losses.update(loss, imgs.shape[0])

                    pbar.set_postfix({'loss_seg': '{0:.4f}'.format(losses_seg.val),
                                      'loss_regression': '{0:.4f}'.format(losses_regression.val),
                                      'loss_embedding': '{0:.4f}'.format(losses_embedding.val),
                                      'loss':'{0:.4f}'.format(losses.val),
                                      'lr':'{0:.5f}'.format(self.optimizer.param_groups[0]['lr'])
                                      })
                    pbar.update(1)

            self.log(' * Train Epoch: {epoch:n}, '
                     'LearningRate {lr:.5f}, SegLoss {losses_seg.avg:.4f}, RegressionLoss {losses_regression.avg:.4f}, '
                     'EmbeddingLoss {losses_embedding.avg:.4f}, '
                     'Loss {losses.avg:.4f}'.format(epoch=self.epoch + 1,
                                                    lr=self.optimizer.param_groups[0]['lr'],
                                                    losses_seg=losses_seg,
                                                    losses_regression=losses_regression,
                                                    losses_embedding=losses_embedding,
                                                    losses=losses))

            if (self.epoch + 1) % 1 == 0:
                model_dir = os.path.join(self.exp_dir, site_name)
                os.makedirs(model_dir, exist_ok=True)
                self.save_model(filename=os.path.join(model_dir,'Epoch_%d_Dice_%.4f' % (self.epoch + 1, val_dice)))


    def update_model(self, inputs, targets):
        out_segs, out_tanhs = self.forward(inputs)
        loss_seg, loss_regression, loss_embedding, loss = self.criterion(out_segs, out_tanhs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_seg.detach(), loss_regression.detach(), loss_embedding.detach(), loss.detach()


    def criterion(self, out_segs, out_tanhs, targets):
        _, _, loss_seg = super(SegNormalNN_CISR, self).criterion(out_segs, targets)

        with torch.no_grad():
            gt_dis = compute_sdf(targets.cpu().numpy())
            gt_dis = torch.from_numpy(gt_dis).float().cuda()
        loss_regression_fn = nn.MSELoss()
        loss_regression = loss_regression_fn(out_tanhs, gt_dis)


        self.embedding.eval()
        with torch.no_grad():
            target_codes, _ = self.embedding(targets)
        seg_codes, _ = self.embedding(out_segs)
        loss_embedding_fn = nn.MSELoss()
        loss_embedding = loss_embedding_fn(seg_codes, target_codes)

        loss = loss_seg + self.beta * loss_regression + self.beta1 * loss_embedding
        return loss_seg, loss_regression, loss_embedding, loss


