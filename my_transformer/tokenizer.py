import os
import re
from pathlib import Path

import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe
from torch.utils import data


def read_data_fra(data_path)->str:
    with open(data_path,'r',encoding='utf-8') as f:
        return f.read()
def preprocess_fra(text:str)->str:
    # Delete extra space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert Spaces between words and ',.!?'
    text = re.sub(r'(\w)([,.!?])', r'\1 \2', text)
    text = re.sub(r'([,.!?])(\w)', r'\1 \2', text)
    return text
def split_line(text:str, num_examples:int)->(list[str], list[str]):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0])
            target.append(parts[1])
    return source, target

class Tokenizer:
    # Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):

        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        self.special_tokens = {"<|begin_of_text|>":num_base_tokens+0,
                               "<|end_of_text|>":num_base_tokens+1,
                               "<|pad|>":num_base_tokens+2}
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.n_words: int = self.model.n_vocab
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    def encode(self,s:str,bos:bool=False,eos:bool=False,
               allowed_special = "all",
               disallowed_special= "all")->list[int]:
        assert type(s) is str
        # See the official docs for details. Or ask me.
        t=self.model.encode(s,allowed_special=allowed_special, disallowed_special=disallowed_special)
        if bos:t.insert(0, self.bos_id)
        if eos:t.append(self.eos_id)
        return t
    def decode(self,t:list[int])->str:
        return self.model.decode(t)

def truncate_pad(line:list[int], num_steps:int, padding_token:list[int])->list[int]:
    if len(line) > num_steps:
        return line[:num_steps]  # truncate
    return line + padding_token * (num_steps - len(line))  # pad

def build_array_fra(lines:list[str], tokenizer:Tokenizer, num_steps:int):
    #Tokenize and build array
    #return Tensor(Tensor(int)),Tensor[int]
    lines = [tokenizer.encode(l,bos=False,eos=True) for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, tokenizer.encode("<|pad|>")) for l in lines])
    valid_len = (array != tokenizer.encode("<|pad|>")[0]).type(torch.int32).sum(1)
    return array, valid_len

def load_data_fra(data_path:str,model_path:str,batch_size, num_steps, num_examples=600):
    #return an iterator
    assert os.path.isfile(model_path), model_path
    assert os.path.isfile(data_path), data_path
    text = preprocess_fra(read_data_fra(data_path))# all text in one string
    source, target = split_line(text, num_examples)#eng, fra sentences
    tokenizer = Tokenizer(model_path)

    src_array, src_valid_len = build_array_fra(source, tokenizer, num_steps)#Tensor(Tensor(int)),Tensor[int]
    tgt_array, tgt_valid_len = build_array_fra(target, tokenizer, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    def load_array(data_arrays, batch_size, is_train=True):
        #Construct a PyTorch data iterator.
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, tokenizer #each data_iter has shape of ((batch_size,num_steps),(batch_size)) * 2


# model_path="/home/pengweixuan/Documents/mylearning/Mytransformer/non_code_files/tokenizer.model"
# raw_text=read_data_fra()
# text1 = preprocess_fra(raw_text)
# print(text1[:80])
# source, target = tokenize_fra(text1)
# print(source[1000:1006], target[1000:1006])
# t=Tokenizer(model_path)
# a=truncate_pad(t.encode(source[0]),10,t.encode("<|pad|>"))
# print(a)
# train_iter, tokenizer = load_data_fra(model_path,batch_size=2, num_steps=8)
# for batch in train_iter:
#     X, X_valid_len, Y, Y_valid_len = [x for x in batch]
#     print( X.type(torch.int32))
#     print(X_valid_len)
#     print( Y.type(torch.int32))
#     print( Y_valid_len)
#     break