import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # print('c_in:', c_in, ' d_model:', d_model, ) c_in : 55 / d_model : 512 for MSL data
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# MultiScaleTokenEmbedding 클래스 새로 정의
class MultiScaleTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_sizes=[3, 5, 7]):
        print('Vic-K_size: ', kernel_sizes)
        super(MultiScaleTokenEmbedding, self).__init__()
        self.kernel_sizes = kernel_sizes
        
        # 각 커널 사이즈에 맞는 Conv1d 레이어 리스트
        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_channels=c_in, out_channels=d_model,
                      kernel_size=k, padding=(k - 1) // 2, padding_mode='circular', bias=False)
            for k in kernel_sizes
        ])
        
        # 각 컨볼루션 출력을 합치기 위한 프로젝션 레이어
        self.projection = nn.Linear(len(kernel_sizes) * d_model, d_model)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print('Vic 1- x.shape: ', x.shape) // torch.Size([256, 100, 55]) for MSL data
        x_permuted = x.permute(0, 2, 1)
        conv_outputs = []
        for conv in self.conv_list:
            # print('Vic 2- x_permuted.shape: ', x_permuted.shape) // torch.Size([256, 55, 100])
            #a = conv(x_permuted)
            #print('Vic 4- a.shape: ', a.shape) //torch.Size([256, 512, 100])
            conv_outputs.append(conv(x_permuted))
        
        # 모든 컨볼루션 결과를 채널 차원에서 결합 (concatenate)
        x_cat = torch.cat(conv_outputs, dim=1).transpose(1, 2)
        # print('Vic 3- x_cat.shape: ', x_cat.shape) // torch.Size([256, 100, 1536])
        # 최종 d_model 차원으로 프로젝션
        x = self.projection(x_cat)
        #print('Vic 5- x.shape: ', x.shape) //torch.Size([256, 100, 512])
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        #self.value_embedding = MultiScaleTokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
