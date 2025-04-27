import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Router import Router
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Forward(nn.Module):
    def __init__(self):
        super(Forward, self).__init__()
    def forward(self, x):
        return x

class feedForward(nn.Module):
    def __init__(self, d_ff, d_model):
        super(feedForward, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))  # (1284,512,7)
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        return x

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)#[4,321,6,512] [4,321,1,512]->[4,321,7,512]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):#(3284,7,512),(4,325,512)
        # 生成形状与 x 相同的全零张量
        # cross = torch.zeros_like(cross)
        # 生成形状与 x 相同的正态分布随机张量
        # x = torch.randn_like(x)
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

class layer(nn.Module):
    def __init__(self,self_attention, cross_attention, d_model, d_ff, num_cells, num_out_path):
        super(layer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.num_cells = num_cells
        self.num_out_path = num_out_path
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.eps = 1e-8
        self.dropout = nn.Dropout(0.1)
        self.router1 = Router(num_out_path, d_model)
        self.router2 = Router(num_out_path, d_model)
        self.forw = Forward()
        self.router3 = Router(num_out_path, d_model)


    def forward(self, x, cross, x_mask, cross_mask, tau, delta):
        B, L, D = cross.shape  # (4,325,512)
        path_prob = [None] * self.num_cells
        cell_output = [None] * self.num_cells
        xor = x
        cell_output[0] = self.dropout(self.self_attention(  # (1284,7,512) (1284,3)
            xor, xor, xor,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        path_prob[0] = self.router1(cross)
        x_glb_ori = xor[:, -1, :].unsqueeze(1)  # (1284,1,512)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # (4,321,512)
        x_glb_attn = self.dropout(self.cross_attention(  # (4,321,512)
            x_glb, cross, cross,  # (4,321,512),(4,325,512)
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,  # (1284,1,512)
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn  # (1284,1,512)
        x_glb = self.norm1(x_glb)  # (1284,1,512)
        cell_output[1] = torch.cat([x[:, :-1, :], x_glb], dim=1) # （1284,7,512）
        path_prob[1] = self.router2(cross)
        cell_output[2] = self.dropout(self.forw(xor))
        path_prob[2] = self.router3(cross)
        if self.num_out_path == 1:
            res1 = 0
            for j in range(self.num_cells):
                res1 += path_prob[j].unsqueeze(-1) * cell_output[j]
            res1 = self.norm2(res1)
            return res1
        else:
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            res_lst = []
            for i in range(self.num_out_path):
                res1 = 0
                for j in range(self.num_cells):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res1 = res1 + cur_path * cell_output[j]
                res_lst.append(res1)
            return res_lst
class layer2(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model,d_ff,num_cells, num_out_path):
        super(layer2, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.num_cells = num_cells
        self.num_out_path = num_out_path
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.router1 = Router(num_out_path, d_model)
        self.router2 = Router(num_out_path, d_model)
        self.forw = Forward()
        self.router3 = Router(num_out_path, d_model)
        self.eps = 1e-8
        self.dropout = nn.Dropout(0.1)

    def forward(self, list, cross, x_mask, cross_mask, tau, delta):
        input1 = list[0]
        input2 = list[1]
        input3 = list[2]

        B, L, D = cross.shape  # (4,325,512)
        path_prob = [None] * self.num_cells
        cell_output = [None] * self.num_cells

        cell_output[0] = self.dropout(self.self_attention(  # (1284,7,512) (1284,3)
            input1, input1, input1,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        path_prob[0] = self.router1(input1)
        x_glb_ori = input2[:, -1, :].unsqueeze(1)  # (1284,1,512)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # (4,321,512)
        x_glb_attn = self.dropout(self.cross_attention(  # (4,321,512)
            x_glb, cross, cross,  # (4,321,512),(4,325,512)
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,  # (1284,1,512)
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn  # (1284,1,512)
        x_glb = self.norm1(x_glb)  # (1284,1,512)
        cell_output[1] = torch.cat([input2[:, :-1, :], x_glb], dim=1) # （1284,7,512）
        path_prob[1] = self.router2(input2)
        cell_output[2] = self.forw(input3)
        path_prob[2] = self.router3(input3)

        if self.num_out_path == 1:
            res1 = 0
            for j in range(self.num_cells):
                res1 += path_prob[j].unsqueeze(-1) * cell_output[j]
            res1 = self.norm2(res1)
            return res1
        else:
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
            res_lst = []
            for i in range(self.num_out_path):
                res1 = 0
                for j in range(self.num_cells):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res1 = res1 + cur_path * cell_output[j]
                res_lst.append(res1)
            return res_lst

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.num_cells = 3
        self.layer1 = layer(self_attention, cross_attention, d_model, d_ff, self.num_cells, self.num_cells)
        self.layer2 = layer2(self_attention, cross_attention, d_model,d_ff, self.num_cells, 1)

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):#(1284,7,512),(4,325,512)
        list = self.layer1(x, cross, x_mask, cross_mask, tau, delta)
        res = self.layer2(list, cross, x_mask, cross_mask, tau, delta)
        y = x = res + x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) #(1284,512,7)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm1(x + y) #(1284,7,512)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1)) #(1284,7,512) 321
        ex_embed = self.ex_embedding(x_enc, x_mark_enc) #(4,325,512) #(4,325,512)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):#(4,96,321),(4,96,4)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None