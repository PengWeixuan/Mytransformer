import os
import re

import torch
from torch.utils import data
import sentencepiece as spm


def read_data_fra(data_path:str)->str:
    assert os.path.isfile(data_path), data_path
    with open(data_path,'r',encoding='utf-8') as f:
        return f.read()
def preprocess_fra(text:str)->str:
    # Delete extra space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert Spaces between words and ',.!?'
    text = re.sub(r'(\w)([,.!?])', r'\1 \2', text)
    text = re.sub(r'([,.!?])(\w)', r'\1 \2', text)
    return text
def split_line(text:str, num_examples:int=None)->(list[str], list[str]):
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
    def __init__(self, model_path: str, data_path:str=None):
        if not os.path.isfile(model_path):
            assert os.path.isfile(data_path), "You need to specify your dataset to train the tokenizer."
            self._train_tokenizer(model_path,data_path)

        self.model = spm.SentencePieceProcessor(model_file=model_path)
        assert self.model.vocab_size() == self.model.GetPieceSize()

        self.n_words: int = self.model.vocab_size()
        self.bos_id: int = self.model.bos_id()
        self.eos_id: int = self.model.eos_id()
        self.pad_id: int = self.model.pad_id()

    def _train_tokenizer(self,model_path,data_path):
        #Train your own tokenizer based on your dataset.
        assert os.path.isfile(data_path), data_path

        vocab_size = 1000#hyper-parameter
        model = model_path[:-6]#delete ".model" in model_path
        model_type = 'bpe'
        #  https://github.com/google/sentencepiece/blob/master/doc/options.md
        spm.SentencePieceTrainer.Train(input=data_path, model_prefix=model, vocab_size=vocab_size,
                                       model_type=model_type, pad_id=0, unk_id=1, eos_id=2, bos_id=3,
                                       byte_fallback=True)

    def encode(self,s:str,bos:bool=False,eos:bool=False)->list[int]:
        t=self.model.Encode(s,out_type=int,add_bos=bos,add_eos=eos)
        return t
    def decode(self,t:list[int])->str:
        return self.model.decode(t)#you can call decode() to call Decode().

# class Tokenizer2:
#     # Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
#     pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501
#
#     def __init__(self, model_path: str):
#
#         assert os.path.isfile(model_path), model_path
#
#         mergeable_ranks = load_tiktoken_bpe(model_path)
#         num_base_tokens = len(mergeable_ranks)
#         self.special_tokens = {"<|begin_of_text|>":num_base_tokens+0,
#                                "<|end_of_text|>":num_base_tokens+1,
#                                "<|pad|>":num_base_tokens+2}
#         self.model = tiktoken.Encoding(
#             name=Path(model_path).name,
#             pat_str=self.pat_str,
#             mergeable_ranks=mergeable_ranks,
#             special_tokens=self.special_tokens,
#         )
#         self.n_words: int = self.model.n_vocab
#         self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
#         self.eos_id: int = self.special_tokens["<|end_of_text|>"]
#         self.pad_id: int = self.special_tokens["<|pad|>"]
#     def encode(self,s:str,bos:bool=False,eos:bool=False,
#                allowed_special = "all",
#                disallowed_special= "all")->list[int]:
#         assert type(s) is str
#         # See the official docs for details. Or ask me.
#         t=self.model.encode(s,allowed_special=allowed_special, disallowed_special=disallowed_special)
#         if bos:t.insert(0, self.bos_id)
#         if eos:t.append(self.eos_id)
#         return t
#     def decode(self,t:list[int])->str:
#         return self.model.decode(t)

def truncate_pad(line:list[int], num_steps:int, padding_token:int)->list[int]:
    if len(line) > num_steps:
        return line[:num_steps]  # truncate
    return line + [padding_token] * (num_steps - len(line))  # pad

def build_array_fra(lines:list[str], tokenizer:Tokenizer, num_steps:int):
    #Tokenize and build array
    #return Tensor(Tensor(int)),Tensor[int]
    lines = [tokenizer.encode(l,bos=False,eos=True) for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, tokenizer.pad_id) for l in lines])
    valid_len = (array != tokenizer.pad_id).type(torch.int32).sum(1)
    return array, valid_len

def load_data_fra(data_path:str, tokenizer_path:str, batch_size, num_steps, num_examples=600):
    #return an iterator
    text = preprocess_fra(read_data_fra(data_path))# all text in one string
    source, target = split_line(text, num_examples)#eng, fra sentences
    tokenizer = Tokenizer(tokenizer_path,data_path)

    src_array, src_valid_len = build_array_fra(source, tokenizer, num_steps)#Tensor(Tensor(int)),Tensor[int]
    tgt_array, tgt_valid_len = build_array_fra(target, tokenizer, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    def load_array(data_arrays, batch_size, is_train=True):
        #Construct a PyTorch data iterator.
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, tokenizer #each data_iter has shape of ((batch_size,num_steps),(batch_size)) * 2

# """TEST"""
# torch.manual_seed(1)
# tokenizer_path= "/home/pengweixuan/Documents/mylearning/Mytransformer/non_code_files/tokenizer.model"
# data_path="../non_code_files/fra.txt"
# raw_text=read_data_fra(data_path)
# text1 = preprocess_fra(raw_text)
# print(text1[:80])
# source, target = split_line(text1)
# print(source[1000:1006], target[1000:1006])
# t=Tokenizer(tokenizer_path,data_path)
# a=truncate_pad(t.encode(source[0]),10,t.pad_id)
# print(a)
# train_iter, tokenizer = load_data_fra(data_path,tokenizer_path, batch_size=2, num_steps=8)
# for batch in train_iter:
#     X, X_valid_len, Y, Y_valid_len = [x for x in batch]
#     print( X.type(torch.int32))
#     print(X_valid_len)
#     print( Y.type(torch.int32))
#     print( Y_valid_len)
#     break
# from collections import Counter
# a=t.encode(text1)
# counter = Counter(a)
# c=counter.most_common()
# a=[]
# for i in c:
#     a.append(i)
#     if i[1]<1000:
#         break
# print(len(a),'/',len(c))