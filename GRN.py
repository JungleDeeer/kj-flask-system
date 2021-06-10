"""

CRN 9.84m

"""
import torch
import torch.nn as nn
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class GRN(nn.Module):
    def __init__(self):
        super(GRN, self).__init__()
        # Main Encoder Part
        self.dilaconv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,dilation=(1,1),padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, dilation=(1,1),padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,  dilation=(1,2),padding=(2,4)),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, dilation=(1,4),padding=(2,8)),
            nn.ELU(),
        )
        self.conv1d = nn.Conv1d(kernel_size=1,dilation=1,out_channels=256,in_channels=4128)
        # self.glu_ = convblocks(1,128)
        self.glus_0 = nn.ModuleList([convblocks(2**i,256) for i in range(6)])
        self.glus_1 = nn.ModuleList([convblocks(2**i,256) for i in range(6)])
        self.glus_2 = nn.ModuleList([convblocks(2**i,256) for i in range(6)])

        self.conv1d_3 = nn.Sequential(nn.Conv1d(kernel_size=1, dilation=1, out_channels=256, in_channels=256),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU()
                                      )
        self.conv1d_4 =nn.Sequential(nn.Conv1d(kernel_size=1, dilation=1, out_channels=128, in_channels=256),
                                      nn.BatchNorm1d(128),
                                      )
        self.conv1d_5 = nn.Sequential(nn.Conv1d(kernel_size=1, dilation=1, out_channels=129, in_channels=128),
                                      nn.BatchNorm1d(129),
                                      nn.Sigmoid()
                                      )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x_list = []
        x = self.dilaconv(x)
        x = x.permute(0,2,1,3)
        x = x.reshape(x.size()[0],x.size()[1],-1)#T×4128
        # print(x.shape)
        x = self.conv1d(x.permute(0,2,1))
        # x = self.glu_(x,x_list)
        # h = 0
        # x = self.glus_0[0](x)
        # h = h+x
        # x = self.glus_0[1](x)
        # h = h +x
        # x = self.glus_0[2](x)
        # t = h + x
        for id in range(6):
            x, h= self.glus_0[id](x)
            x_list.append(h)
            # x_list.append(x)
        for id in range(6):
            x ,h = self.glus_1[id](x)
            x_list.append(h)
        for id in range(6):
            x ,h = self.glus_2[id](x)
            x_list.append(h)
            x_list.append(x)
        t=0
        for i in x_list:
            t = t + i

        t = self.conv1d_3(t)

        t = self.conv1d_4(t)

        t = self.conv1d_5(t)
        # print(t.shape)
        return t.permute(0, 2, 1)
        # x = x.reshape(x.size()[0], x.size()[1], -1)
        # x, _ = self.lstm(x)
        # x = x.reshape(x.size()[0], x.size()[1], 256, -1)
        # x = x.permute(0, 2, 1, 3)
        # x = self.de(x, s)
        # return x.squeeze()


class convblocks(nn.Module):
    def __init__(self,dilation,input):
        super(convblocks, self).__init__()

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(kernel_size=1,dilation=1,out_channels=64,in_channels=input),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.dilated1 = nn.Sequential(nn.Conv1d(kernel_size=7,out_channels=64,in_channels=64,dilation=dilation,padding=dilation*3),
                        nn.BatchNorm1d(64),
                        nn.Sigmoid())
        self.dilated2 = nn.Conv1d(kernel_size=7, out_channels=64, in_channels=64, dilation=dilation,padding=dilation*3)
        self.conv1d_2 = nn.Sequential(nn.Conv1d(kernel_size=1,dilation=1,in_channels=64,out_channels=256),
                                      nn.BatchNorm1d(256)
                                      )
        self.ELU = nn.ELU()
    def forward(self,x):
        t = x

        x = self.conv1d_1(x)
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)

        x = x1 *x2
        # print(x.shape)
        x = self.conv1d_2(x)
        h = x
        x = x +t
        x = self.ELU(x)
        return x,h
# class Conv1(nn.Module):
#     def __init__(self):
#         super(Conv1, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9, stride=1),
#             nn.BatchNorm2d(32),
#             nn.ELU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ELU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, stride=1),
#             nn.BatchNorm2d(64),
#             nn.ELU()
#         )
#         self.maxpooling = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.maxpooling(x)
#         x = self.conv2(x)
#         x = self.maxpooling(x)
#         return x

# class DilatedConv1(nn.Module):
#     def __init__(self):
#         super(DilatedConv1, self).__init__()
#         self.conv_d1 = nn.Sequential(nn.Conv1d(kernel_size=3,dilation=2,out_channels=16,in_channels=256),
#                                       #nn.BatchNorm2d(256),
#                                       nn.ELU())
#         self.conv_d2 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=4, out_channels=16, in_channels=16),
#                                      # nn.BatchNorm2d(256),
#                                      nn.ELU())
#         self.conv_d3 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=8, out_channels=16, in_channels=16),
#                                      # nn.BatchNorm2d(256),
#                                      nn.ELU())
#         self.conv_d4 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=16, out_channels=16, in_channels=16),
#                                      # nn.BatchNorm2d(256),
#                                      nn.ELU())
#         self.conv_d5 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=32, out_channels=16, in_channels=16),
#                                      # nn.BatchNorm2d(256),
#                                      nn.ELU())
#         self.conv_d6 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=64, out_channels=16, in_channels=16),
#                                      # nn.BatchNorm2d(256),
#                                      nn.ELU())
#         self.conv_d7 = nn.Sequential(nn.Conv1d(kernel_size=3, dilation=128, out_channels=16, in_channels=16),
#                                      # nn.BatchNorm2d(256),
#                                      nn.ELU())
#     def forward(self,x):
#         x = self.conv_d1(x)
#         x = self.conv_d2(x)
#         x = self.conv_d3(x)
#         x = self.conv_d4(x)
#         x = self.conv_d5(x)
#         x = self.conv_d6(x)
#         x = self.conv_d7(x)
#         return x
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.en1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(16),
#             nn.ELU()
#             )
#         self.en2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(32),
#             nn.ELU()
#         )           # 512x32
#         self.en3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(64),
#             nn.ELU()
#         )
#         self.en4 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(128),
#             nn.ELU()
#         )
#         self.en5 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(256),
#             nn.ELU()
#         )
#
#     def forward(self, x):
#         en_list = []
#         x = self.en1(x)
#         en_list.append(x)
#         # print(x.size())
#         x = self.en2(x)
#         en_list.append(x)
#         # print(x.size())
#         x = self.en3(x)
#         en_list.append(x)
#         # print(x.size())
#         x = self.en4(x)
#         en_list.append(x)
#         # print(x.size())
#         x = self.en5(x)
#         en_list.append(x)
#         # print(x.size())
#         return x, en_list

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.de5 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(128),
#             nn.ELU()
#         )
#         self.de4 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(64),
#             nn.ELU()
#         )
#         self.de3 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(32),
#             nn.ELU()
#         )
#         self.de2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1)),
#             nn.BatchNorm2d(16),
#             nn.ELU()
#         )
#         self.de1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2)),
#             nn.BatchNorm2d(1),
#             nn.ELU()
#         )
#
#     def forward(self, x, x_list):
#         x = self.de5(torch.cat((x, x_list[-1]), dim=1))
#         # print(x.size())
#         x = self.de4(torch.cat((x, x_list[-2]), dim=1))
#         # print(x.size())
#         x = self.de3(torch.cat((x, x_list[-3]), dim=1))
#         # print(x.size())
#         x = self.de2(torch.cat((x, x_list[-4]), dim=1))
#         # print(x.size())
#         x = self.de1(torch.cat((x, x_list[-5]), dim=1))
#         # print(x.size())
#         return x
#

# class LSTM_1(nn.Module):
#     def __init__(self):
#         super(LSTM_1, self).__init__()
#         self.lstm_input = nn.LSTM(input_size=11 * 161, hidden_size=1024, batch_first=True)
#         self.lstm1 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm3 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm4 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm_output = nn.LSTM(input_size=1024, hidden_size=161, batch_first=True)
#
#     def forward(self, x):
#         x = self.lstm_input(x)
#         x = self.lstm1(x)
#         x = self.lstm2(x)
#         x = self.lstm3(x)
#         x = self.lstm4(x)
#         x = self.lstm_output(x)
#
#         return x
#
#
# class LSTM_2(nn.Module):
#     def __init__(self):
#         super(LSTM_2, self).__init__()
#         self.lstm_input = nn.LSTM(input_size=161, hidden_size=1024, batch_first=True)
#         self.lstm1 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm3 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm4 = nn.LSTM(input_size=1024, hidden_size=1024, batch_first=True)
#         self.lstm_output = nn.LSTM(input_size=1024, hidden_size=161, batch_first=True)
#
#     def forward(self, x):
#         x = self.lstm_input(x)
#         x = self.lstm1(x)
#         x = self.lstm2(x)
#         x = self.lstm3(x)
#         x = self.lstm4(x)
#         x = self.lstm_output(x)
#
#         return x


