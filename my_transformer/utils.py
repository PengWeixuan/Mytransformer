import torch
from torch import nn


def sequence_mask(x, valid_len, value=0.0):
    maxlen = x.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=x.device)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x


def masked_softmax(X, valid_lens):
    #X.shape==[batchsize,n_heads,seqlen,seqlen]
    assert X.dim() == 4
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[2])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# a=sequence_mask(X, torch.tensor([1, 2]))
# X = torch.ones(2, 3, 4)
# b=sequence_mask(X, torch.tensor([1, 2]), value=-1)
#
# c=masked_softmax(torch.rand(1,2, 2, 4), torch.tensor([2, 3]))
# d=masked_softmax(torch.rand(1,2, 2, 4), torch.tensor([[1, 3], [2, 4]]))
# print(a)
# print(b)
# print(c)
# print(d)
