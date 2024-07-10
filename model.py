import torch
from torch import nn

num_hiddens = 256

class EBD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EBD, self).__init__(*args, **kwargs)
        self.word_ebd = nn.Embedding(29, num_hiddens)
        self.pos_ebd = nn.Embedding(12, num_hiddens)
        self.pos_t = torch.arange(0, 12).reshape(1, 12)
    
    # X: (batch_size, length)
    def forward(self, X:torch.tensor):
        
        return self.word_ebd(X) + self.pos_ebd(self.pos_t[:, :X.shape[-1]].to(X.device))

def attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, M:torch.Tensor):
    A = Q @ K.transpose(-1, -2) / (Q.shape[-1] ** 0.5)
    M = M.unsqueeze(1)
    A.masked_fill_(M==0, -torch.tensor(float('inf')))
    A = torch.softmax(A, dim=-1)
    O = A @ V
    return O

def transpose_o(O):
    O = O.transpose(-2, -3)
    O = O.reshape(O.shape[0], O.shape[1], -1)
    return O

def transpose_qkv(QKV:torch.tensor):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4, QKV.shape[-1]//4)
    QKV = QKV.transpose(-2, -3)
    return QKV

class Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Attention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=False)
    def forward(self, X, M:torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, M)
        O = transpose_o(O)
        O = self.Wo(O)
        return O

class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddNorm, self).__init__(*args, **kwargs)
        self.add_norm = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, X, X1):
        X1 = self.add_norm(X1)
        X = X + X1
        X = self.dropout(X)
        return X

class Pos_FFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Pos_FFN, self).__init__(*args, **kwargs)
        self.lin_1 = nn.Linear(num_hiddens, 1024, bias=False)
        self.relu1 = nn.ReLU()
        self.lin_2 = nn.Linear(1024, num_hiddens, bias=False)
        self.relu2 = nn.ReLU()
    
    def forward(self, X):
        X = self.lin_1(X)
        X = self.relu1(X)
        X = self.lin_2(X)
        X = self.relu2(X)
        return X

class Encoder_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder_block, self).__init__(*args, **kwargs)
        self.attention = Attention_block()
        self.add_norm_1 = AddNorm()
        self.FFN = Pos_FFN()
        self.add_norm_2 = AddNorm()
    def forward(self, X, I_m):
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X, I_m)
        X = self.add_norm_1(X, X_1)
        X_1 = self.FFN(X)
        X = self.add_norm_2(X, X_1)
        return X

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        self.ebd = EBD()
        self.encoder_blks = nn.Sequential()
        self.encoder_blks.append(Encoder_block())
        self.encoder_blks.append(Encoder_block())
        self.encoder_blks.append(Encoder_block())
        self.encoder_blks.append(Encoder_block())
    
    def forward(self, X, I_m):
        X = self.ebd(X)
        for encoder_blk in self.encoder_blks:
            X = encoder_blk(X, I_m)
        return X

class CrossAttention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossAttention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=False)
    def forward(self, X, X_en, I_M):
        Q, K, V = self.Wq(X), self.Wk(X_en), self.Wv(X_en)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, I_M)
        O = transpose_o(O)
        O = self.Wo(O)
        return O

class Decoder_blk(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder_blk, self).__init__(*args, **kwargs)
        self.attention = Attention_block()
        self.add_norm_1 = AddNorm()
        self.cross_attention = CrossAttention_block()
        self.add_norm_2 = AddNorm()
        self.FFN = Pos_FFN()
        self.add_norm_3 = AddNorm()
        mask_matrix = torch.ones(12, 12)
        self.tril_mask = torch.tril(mask_matrix).unsqueeze(0)
        
    def forward(self, X_t, O_m, X_en, I_m):
        O_m = O_m.unsqueeze(-2)
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X_t, O_m * self.tril_mask[:, :O_m.shape[-1], :O_m.shape[-1]].to(X_t.device))
        X_t = self.add_norm_1(X_t, X_1)
        X_1 = self.cross_attention(X_t, X_en, I_m)
        X_t = self.add_norm_2(X_t, X_1)
        X_1 = self.FFN(X_t)
        X_t = self.add_norm_3(X_t, X_1)
        return X_t

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        self.ebd = EBD()
        self.decoder_blks = nn.Sequential()
        self.decoder_blks.append(Decoder_blk())
        self.decoder_blks.append(Decoder_blk())
        self.decoder_blks.append(Decoder_blk())
        self.decoder_blks.append(Decoder_blk())
        self.dense = nn.Linear(num_hiddens, 29, bias=False)
    
    def forward(self, X_t, O_m, X_en, I_m):
        X_t = self.ebd(X_t)
        for layer in self.decoder_blks:
            X_t = layer(X_t, O_m, X_en, I_m)
        X_t = self.dense(X_t)
        return X_t

class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, X_s, I_m, X_t, O_m):
        X_en = self.encoder(X_s, I_m)
        X = self.decoder(X_t, O_m, X_en, I_m)
        return X

if __name__ == "__main__":
    aaa = torch.ones((2, 12)).long()
    bbb = torch.ones((2, 4)).long()
    my_model = Transformer()
    o = my_model(aaa, bbb)
    
    pass
