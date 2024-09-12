import collections
import os

import torch
from torch.utils import data


def read_data_fra(data_path:str)->str:
    assert os.path.isfile(data_path), data_path
    with open(data_path,'r',encoding='utf-8') as f:
        return f.read()
def preprocess_fra(text:str)->str:
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_fra(text, num_examples:int=None)->(list[list[str]], list[list[str]]):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def count_corpus(tokens):
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def truncate_pad(line:list[int], num_steps:int, padding_token:int)->list[int]:
    if len(line) > num_steps:
        return line[:num_steps]  # truncate
    return line + [padding_token] * (num_steps - len(line))  # pad

def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_fra(data_path:str,batch_size, num_steps, num_examples=600):
    #return an iterator
    text = preprocess_fra(read_data_fra(data_path))# all text in one string
    source, target = tokenize_fra(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    def load_array(data_arrays, batch_size, is_train=True):
        #Construct a PyTorch data iterator.
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab #each data_iter has shape of ((batch_size,num_steps),(batch_size)) * 2


# data_path="../non_code_files/fra.txt"
# raw_text=read_data_fra(data_path)
# text1 = preprocess_fra(raw_text)
# print(text1[:80])
# source, target = tokenize_fra(text1)
# print(source[1000:1006], target[1000:1006])
# src_vocab = Vocab(source, min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
# a=truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
# print(a)
# train_iter, src_vocab, tgt_vocab = load_data_fra(data_path,batch_size=2, num_steps=8)
#
# for batch in train_iter:
#     X, X_valid_len, Y, Y_valid_len = [x for x in batch]
#     print( X.type(torch.int32))
#     print(X_valid_len)
#     print( Y.type(torch.int32))
#     print( Y_valid_len)
#     break