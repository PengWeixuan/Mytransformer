import torch
from torch import nn


def sequence_mask(x, valid_len, value=0.0):
    #x.dim()==2
    maxlen = x.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(X, valid_lens):
    #X.shape==[batchsize,n_heads,seqlen,seqlen], valid_lens.shape==[batchsize] or [batchsize,seqlen]
    #assert X.dim() == 4
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[2])
        else:# In this case valid_lens.dim() == 2
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# Totally SHIT code copied from "DIVE INTO DEEP LEARNING". Implement your own mask if you need.
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# a=sequence_mask(X, torch.tensor([1, 2]))
# X = torch.ones(2, 3, 4)
# b=sequence_mask(X, torch.tensor([1, 2]), value=-1)
# X=torch.rand(2, 5, 4, 6)
# valid_lens=torch.tensor([2, 3])
# valid_lens=torch.repeat_interleave(valid_lens, repeats=X.shape[1], dim=0)
# c=masked_softmax(X, valid_lens)

# dec_valid_lens = torch.arange(1, X.shape[2] + 1).repeat(X.shape[0], 1)
# dec_valid_lens=torch.repeat_interleave(dec_valid_lens, repeats=X.shape[1], dim=0)
# print(dec_valid_lens.shape)
# d=masked_softmax(X, dec_valid_lens)
# print(a)
# print(b)
# print(c)
# print(d)
