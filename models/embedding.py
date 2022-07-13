#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/22 4:44 PM
# @Author  : Jingyang.Zhang
'''
autoencoder for segmentation embedding
'''
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class embedding(nn.Module):
    def __init__(self):
        # 输入图像需要为 384 大小
        super(embedding, self).__init__()
        self.num_filters = 16
        self.num_channels = 2
        self.zdim = 64
        filters = [
            self.num_filters,
            self.num_filters * 2,
            self.num_filters * 4,
        ]

        # encoder
        self.en_conv1 = conv_block(self.num_channels, filters[0])  # [b, 16, 192, 192]
        self.en_conv2 = conv_block(filters[0], filters[1])  # [b, 32, 96, 96]
        self.en_conv3 = conv_block(filters[1], filters[2])  # [b, 64, 48, 48]
        self.en_conv4 = nn.Sequential(
            nn.Conv2d(filters[2], 1, 3, stride=(3,3), padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)  # [b, 1, 16, 16]
        )
        self.en_fc = nn.Sequential(nn.Linear(256, 64),
                                   nn.ReLU(inplace=True)) # [b, 64]
        # self.en_fc = nn.Linear(256, 64)  # [b, 64]

        # decoder
        self.de_fc = nn.Sequential(nn.Linear(64, 256),
                                   nn.ReLU(inplace=True))  # [b, 1, 16, 16]

        self.de_conv4 = nn.Sequential(
            nn.ConvTranspose2d(1, filters[2], kernel_size=7, stride=(3,3), padding=2),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
        )
        self.de_conv3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
        )
        self.de_conv2 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )
        self.de_conv1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], self.num_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(filters[0]),
            nn.Softmax2d()
        )
        print('Initial acnn.')

    def forward(self,x):
        assert x.shape[1] == 2 and x.shape[2] == 384 and x.shape[3] == 384
        en_conv1 = self.en_conv1(x)
        en_conv2 = self.en_conv2(en_conv1)
        en_conv3 = self.en_conv3(en_conv2)
        en_conv4 = self.en_conv4(en_conv3)

        assert en_conv4.shape[1] == 1 and en_conv4.shape[2] == 16 and en_conv4.shape[3] == 16
        code = self.en_fc(en_conv4.view([en_conv4.shape[0], -1]))

        de_code = self.de_fc(code).view([code.shape[0],1,16,16])
        de_conv4 = self.de_conv4(de_code)
        de_conv3 = self.de_conv3(de_conv4)
        de_conv2 = self.de_conv2(de_conv3)
        de_conv1 = self.de_conv1(de_conv2)

        return code, de_conv1






