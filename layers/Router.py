import torch.nn as nn
import torch
import torch.nn.functional as F

def activateFunc(x):
    x = F.tanh(x)
    return F.relu(x)


class Router(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(nn.Linear(embed_size*2, embed_size),
                                 nn.Linear(embed_size, num_out_path))
    #     self.init_weights()
    #
    # def init_weights(self):
    #     self.mlp[1].bias.data.fill_(1.5)

    def forward(self, x): #(1284, 7, 512),(4,321,512)
        x = x.permute(0, 2, 1)
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out, avg_out],1).permute(0,2,1) #(1284,1,1024)
        x = f_out.contiguous().view(f_out.size(0), -1)
        x = self.mlp(x)
        return activateFunc(x)