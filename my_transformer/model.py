import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from my_transformer.tokenizer import Tokenizer
from my_transformer.train_predict import predict
from my_transformer.utils import masked_softmax

@dataclass
class ModelArgs:
    num_hiddens: int = 32# or dim
    enc_layers: int = 2
    dec_layers: int = 2
    n_heads: int = 4
    vocab_size: int = 128003
    dropout:float=0.1
    ffn_dim_multiplier=2
    bias:bool=True

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
        return self.ln(X + self.dropout(Y))

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Attention,self).__init__()
        self.n_heads=args.n_heads
        self.head_dim = args.num_hiddens // args.n_heads

        self.wq = nn.Linear(args.num_hiddens, self.n_heads * self.head_dim, bias=args.bias)
        self.wk = nn.Linear(args.num_hiddens, self.n_heads * self.head_dim, bias=args.bias)
        self.wv = nn.Linear(args.num_hiddens, self.n_heads * self.head_dim, bias=args.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.num_hiddens, bias=args.bias)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self, xq:torch.Tensor,xk:torch.Tensor,xv:torch.Tensor, valid_lens):
        #xk,xv.shape==[batchsize,seqlen,num_hidden]
        #valid_lens.shape==[batchsize]

        xq, xk, xv = self.wq(xq), self.wk(xk), self.wv(xv)

        # shape [batchsize,seqlen,n_heads,num_hidden/n_heads]
        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
        xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_heads, self.head_dim)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_heads, self.head_dim)

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
        shape=output.shape
        output=output.transpose(1,2).reshape(shape[0], shape[2],-1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super(FeedForward,self).__init__()
        dim=args.num_hiddens
        hidden_dim=int(args.ffn_dim_multiplier*dim)
        self.w1=nn.Linear(dim,hidden_dim,bias=args.bias)
        self.w2=nn.Linear(hidden_dim,dim,bias=args.bias)
    def forward(self,h:torch.Tensor):
        return self.w2(F.relu(self.w1(h)))

class EncoderBlock(nn.Module):
    def __init__(self,layerid,args:ModelArgs):
        super(EncoderBlock,self).__init__()
        self.layerid=layerid
        self.attention=Attention(args)
        self.addnorm1=AddNorm(args)
        self.ffn=FeedForward(args)
        self.addnorm2 = AddNorm(args)

    def forward(self,h:torch.Tensor, valid_lens):
        Y = self.addnorm1(h, self.attention(h,h,h, valid_lens))
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

class DecoderBlock(nn.Module):
    def __init__(self,layerid, args:ModelArgs):
        super(DecoderBlock,self).__init__()
        self.layerid = layerid
        self.masked_attention=Attention(args)
        self.addnorm1 = AddNorm(args)
        self.attention=Attention(args)
        self.addnorm2 = AddNorm(args)
        self.ffn=FeedForward(args)
        self.addnorm3 = AddNorm(args)
    def forward(self,x:torch.Tensor,enc_kv,enc_valid_len):

        if self.training:
            batch_size, seqlen, _ = x.shape
            # upper triangular mask
            dec_valid_lens = torch.arange(1, seqlen + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        h=self.addnorm1(x, self.masked_attention(x,x,x,dec_valid_lens))
        h=self.addnorm2(h, self.attention(x,enc_kv,enc_kv,enc_valid_len))
        h=self.addnorm3(h,self.ffn(h))
        return h


class Decoder(nn.Module):
    def __init__(self,paras:ModelArgs):
        super(Decoder,self).__init__()
        self.paras = paras
        self.num_hiddens = paras.num_hiddens
        self.dropout = paras.dropout
        self.embedding = nn.Embedding(paras.vocab_size, paras.num_hiddens)
        self.layers = torch.nn.ModuleList()
        for i in range(paras.dec_layers):
            self.layers.append(DecoderBlock(i,paras))

        self.output=nn.Linear(paras.num_hiddens,paras.vocab_size,bias=paras.bias)

    def forward(self,x:torch.Tensor,enc_out:torch.Tensor,enc_valid_len):
        h = positionalEncoding(self.embedding(x) * math.sqrt(self.num_hiddens), self.num_hiddens, self.dropout)

        for layer in self.layers:
            h = layer(h, enc_out, enc_valid_len)
        return self.output(h)

class Transformer(nn.Module):
    def __init__(self, paras:ModelArgs):
        super(Transformer, self).__init__()
        self.encoder=Encoder(paras)
        self.decoder=Decoder(paras)
    def forward(self,enc_X,dec_X,enc_valid_len):
        enc_out=self.encoder(enc_X,enc_valid_len)
        return self.decoder(dec_X,enc_out,enc_valid_len)

# if __name__=="__main__":
#     # test
#     model_path="C:\\Users\\39936\\Downloads\\tokenizer.model"
#     tokenizer = Tokenizer(model_path)
#
#     args: ModelArgs=ModelArgs(vocab_size=tokenizer.n_words)
#
#     bsz,seqlen=2,10
#     enc_validlen=torch.tensor([3,2])
#     net=Transformer(args)
#     net.eval()
#     enc_X=torch.ones((bsz,seqlen), dtype=torch.long)
#     dec_X=torch.ones((bsz,3), dtype=torch.long)
#     a=net(enc_X,dec_X,enc_validlen)
#     print(a.shape)
#
#     b=predict(net,"Hello world.",tokenizer,10,enc_X.device)
#     print(b)
