#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 11:27 AM
# @Author  : Jingyang.Zhang
'''
Unet with regression head
'''
import torch
import torch.nn as nn


class Unet_regression(nn.Module):
    def __init__(self):
        super(Unet_regression, self).__init__()
        self.num_filters = 32
        self.num_channels = 3
        self.num_classes = 2
        filters = [self.num_filters,
                   self.num_filters * 2,
                   self.num_filters * 4,
                   self.num_filters * 8,
                   self.num_filters * 16]

        self.conv1 = conv_block(self.num_channels, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv_block(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = conv_block(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(filters[2], filters[3], if_dropout=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.center = conv_block(filters[3], filters[4], if_dropout=True)

        self.up4 = UpCatconv(filters[4], filters[3], if_dropout=True)
        self.up3 = UpCatconv(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])
        self.out_seg = nn.Sequential(nn.Conv2d(filters[0], self.num_classes, kernel_size=1),
                                     nn.Softmax2d())
        self.out_tanh = nn.Sequential(nn.Conv2d(filters[0], 1, kernel_size=1),
                                      nn.Tanh())

        print('Initial Unet_regression.')


    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        center = self.center(pool4)

        up_4 = self.up4(conv4, center)
        up_3 = self.up3(conv3, up_4)
        up_2 = self.up2(conv2, up_3)
        up_1 = self.up1(conv1, up_2)

        out_seg = self.out_seg(up_1)
        out_tanh = self.out_tanh(up_1)


        return out_seg, out_tanh


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, if_dropout=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.if_dropout = if_dropout
        if self.if_dropout == True:
            self.dropout = nn.Dropout2d(0.5)


    def forward(self, x):
        x = self.conv(x)
        if self.if_dropout == True:
            return self.dropout(x)
        else:
            return x


class UpCatconv(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, if_dropout=False):
        super(UpCatconv, self).__init__()

        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, if_dropout=if_dropout)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, if_droput=if_dropout)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        out = self.conv(torch.cat([inputs, outputs], dim=1))

        return out

