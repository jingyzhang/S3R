from __future__ import print_function
import os
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class SegNormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for segmentation
    '''
    def __init__(self, agent_config):

        super(SegNormalNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        self.exp_dir = self.config['exp_dir']
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # Initialization
        self.model = self.create_model()
        self.criterion_fn1 = nn.BCELoss()
        self.criterion_fn2 = dice_loss
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()  # Initialize optimizer with new learning rate


    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # Load model
        if cfg['model_weights'] is not None:
            print('==> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn1 = self.criterion_fn1.cuda()
        # self.criterion_fn2 = self.criterion_fn2.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def learn_batch(self, site_name, train_loader, val_loader=None):
        if self.config['reset_optimizer']:
            self.log('Optimizer is reset!')
            self.init_optimizer()
        else:
            self.optimizer.param_groups[0]['lr'] = self.config['lr']

        for self.epoch in range(self.config['epoches']):
            losses_bce = AverageMeter()
            losses_dice = AverageMeter()
            losses = AverageMeter()

            self.model.train()
            self.scheduler.step()

            with tqdm(total=len(train_loader), desc='Epoch %d/%d' % (self.epoch+1,self.config['epoches']), unit='batch') \
                    as pbar:

                for batch in train_loader:

                    imgs, gts = batch['img'], batch['gt']
                    if self.gpu:
                        imgs, gts = imgs.cuda(), gts.cuda()

                    loss_bce, loss_dice, loss, outs = self.update_model(imgs, gts)
                    imgs = imgs.detach()
                    gts = gts.detach()

                    losses_bce.update(loss_bce, imgs.size(0))
                    losses_dice.update(loss_dice, imgs.size(0))
                    losses.update(loss, imgs.size(0))

                    pbar.set_postfix({'bce_loss': '{0:.4f}'.format(losses_bce.val),
                                      'dice_loss': '{0:.4f}'.format(losses_dice.val),
                                      'loss':'{0:.4f}'.format(losses.val),
                                      'lr':'{0:.5f}'.format(self.optimizer.param_groups[0]['lr'])
                                      })
                    pbar.update(1)

            # print result in each epoch
            self.log(' * Train Epoch: {epoch:n}, '
                     'LearningRate {lr:.5f}, BCELoss {losses_bce.avg:.4f}, DiceLoss {losses_dice.avg:.4f}, '
                     'Loss {losses.avg:.4f}'.format(epoch=self.epoch + 1,
                                                    lr=self.optimizer.param_groups[0]['lr'],
                                                    losses_bce=losses_bce,
                                                    losses_dice=losses_dice,
                                                    losses=losses))

            # save model
            if (self.epoch + 1) % 1 == 0:
                model_dir = os.path.join(self.exp_dir, site_name)
                os.makedirs(model_dir, exist_ok=True)
                self.save_model(filename=os.path.join(model_dir,'Epoch_%d_Dice_%.4f' % (self.epoch + 1, val_dice)))


    def update_model(self, inputs, targets):
        outs = self.forward(inputs)
        loss_bce, loss_dice, loss = self.criterion(outs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_bce.detach(), loss_dice.detach(), loss.detach(), outs

    def criterion(self, preds, targets):
        loss_bce = self.criterion_fn1(preds, targets)
        loss_dice = self.criterion_fn2(preds[:,1,:,:], targets[:,1,:,:])
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        return loss_bce, loss_dice, loss

    def forward(self,x):
        return self.model.forward(x)


    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename + '.pth')
        print('=> Saving model to:', filename + '.pth')



def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss




