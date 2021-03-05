#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:05:26 2021

@author: saikishorehr
"""

import torch
import torchvision.transforms.functional as tf
import torch.nn as nn

class doubleconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(doubleconv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,in_channel,out_channel,features=[64,128,256,512]):
        super(Unet,self).__init__()
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
         
        for feature in features:
            self.downs.append(doubleconv(in_channel,feature))
            in_channel=feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
            self.ups.append(doubleconv(feature*2,feature))

        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.bottleneck=doubleconv(features[-1],features[-1]*2)
        self.final_conv=nn.Conv2d(features[0], out_channel, kernel_size=1)
    def forward(self,x):

        copy_list=[]
        for mod in self.downs:
            x=mod(x)
            copy_list.append(x)
            x=self.pool(x)

        x=self.bottleneck(x)
        copy_list=copy_list[::-1]

        for i in range(0,len(self.ups),2):
            x=self.ups[i](x)
            curr_connection=copy_list[i//2]

            if x.shape != curr_connection.shape:
                x=tf.resize(x, size=curr_connection.shape[2:])

            x=torch.cat((x,curr_connection),dim=1)

            x=self.ups[i+1](x)

        return self.final_conv(x)


def test():
        x = torch.randn((3, 1, 161, 161))
        model = Unet(in_channel=1, out_channel=1)
        preds = model(x)
        assert preds.shape == x.shape

test()


