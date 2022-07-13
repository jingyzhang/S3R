#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/13 10:07 AM
# @Author  : Jingyang.Zhang
'''
'''
import torch
import torch.nn as nn

def apply_dropout(model:nn.Module):
    for opt in model.modules():
        if type(opt) == nn.Dropout2d:
            opt.is_training = True
    return model

def close_dropout(model:nn.Module):
    for opt in model.modules():
        if type(opt) == nn.Dropout2d:
            opt.is_training = False
    return model