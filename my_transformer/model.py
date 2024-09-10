import math

import torch
from torch import nn
import torch.nn.functional as F
from utils import masked_softmax

class ModelArgs:
    num_hiddens: int = 512# or dim
    enc_layers: int = 6
    dec_layers: int = 6
    n_heads: int = 4
    vocab_size: int = 10000
    dropout:float=0
    ffn_dim_multiplier=4

def positionalEncoding(X:torch.Tensor,num_hiddens,dropout):
    # X.shape==[batch_size,seqlen,num_hiddens]
    P=torch.zeros((1,X.shape[1],num_hiddens))
    pos=torch.arange(X.shape[1], dtype=torch.float32).reshape(-1, 1)
    i=torch.pow(10000,torch.arange(0,num_hiddens,2, dtype=torch.float32)/num_hiddens)
    pos_i=pos/i
    P[:,:,0::2]=torch.sin(pos_i)
    P[:,:,1::2]=torch.cos(pos_i)
    X=X+P[:,:,:].to(X.device)
    return F.dropout(X,p=dropout)

class AddNorm(nn.Module):
    def __init__(self, args:ModelArgs):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.ln = nn.LayerNorm(args.num_hiddens)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads=args.n_heads
        self.head_dim = args.num_hiddens // args.n_heads

        self.wq = nn.Linear(args.num_hiddens, self.n_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(args.num_hiddens, self.n_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(args.num_hiddens, self.n_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.num_hiddens, bias=True)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x:torch.Tensor,valid_lens):
        #x.shape==[batchsize,seqlen,num_hidden]
        #valid_lens.shape==[batchsize]
        batchsize, seqlen, _=x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        #xq,xk,xv all with shape of [batchsize,seqlen,n_heads,num_hidden/n_heads]
        xq = xq.reshape(batchsize, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(batchsize, seqlen, self.n_heads, self.head_dim)
        xv = xv.reshape(batchsize, seqlen, self.n_heads, self.head_dim)

        #shape [batchsize,n_heads,seqlen,num_hidden/n_heads]
        #put "batchisize" dim and "n_heads" dim together to take full utilize of parallism
        xq=xq.permute(0, 2, 1, 3)
        xk=xk.transpose(1, 2)
        xv=torch.transpose(xv,1,2)

        # shape [batchsize,n_heads,seqlen,seqlen]
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.n_heads, dim=0)
        attention_weights=masked_softmax(scores,valid_lens)
        #shape [batchsize,n_heads,seqlen,num_hidden]
        output = torch.matmul(self.dropout(attention_weights), xv)
        output=output.transpose(1,2).reshape(batchsize, seqlen,-1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim=args.num_hiddens
        hidden_dim=int(args.ffn_dim_multiplier*dim)
        self.w1=nn.Linear(dim,hidden_dim,bias=True)
        self.w2=nn.Linear(hidden_dim,dim,bias=True)
    def forward(self,h:torch.Tensor):
        return self.w2(F.relu(self.w1(h)))

class EncoderBlock(nn.Module):
    def __init__(self,layerid,args:ModelArgs):
        super(EncoderBlock,self).__init__()
        self.layerid=layerid
        self.attention=Attention(args)
        self.addnorm1=AddNorm(args)
        self.addnorm2 = AddNorm(args)
        self.ffn=FeedForward(args)

    def forward(self,h:torch.Tensor, valid_lens):
        Y = self.addnorm1(h, self.attention(h, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Encoder(nn.Module):
    def __init__(self,paras:ModelArgs):
        super(Encoder,self).__init__()
        self.paras=paras
        self.num_hiddens=paras.num_hiddens
        self.dropout=paras.dropout

        self.embedding = nn.Embedding(paras.vocab_size,paras.num_hiddens)
        self.layers=torch.nn.ModuleList()
        for i in range(paras.enc_layers):
            self.layers.append(EncoderBlock(i,paras))
    def forward(self,x:torch.Tensor, valid_lens):
        h=positionalEncoding(self.embedding(x)* math.sqrt(self.num_hiddens),self.num_hiddens,self.dropout)

        for layer in self.layers:
            h=layer(h, valid_lens)
        return h

class Transformer(nn.Module):
    def __init__(self,encoder,decoder):
        super(Transformer, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,enc_X,dec_X):
        enc_out=self.encoder(enc_X)
        return self.decoder(dec_X,enc_out)

args=ModelArgs()
bsz,seqlen=2,10
validlen=torch.tensor([3,2])
encoder=Encoder(args)
encoder.eval()

a=encoder(torch.ones((bsz,seqlen), dtype=torch.long),validlen)
print(a.shape)
