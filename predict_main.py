import torch

from my_transformer.model import ModelArgs, Transformer
from my_transformer.tokenizer import Tokenizer
from my_transformer.train_predict import predict


def translate(english:str):
    tokenizer_path = "C:\\Users\\39936\\Downloads\\tokenizer.model"

    device = torch.device(f'cuda:0')
    max_output_len=10

    tokenizer = Tokenizer(tokenizer_path)
    args = ModelArgs(vocab_size=tokenizer.n_words)
    net = Transformer(args)
    net.load_state_dict(torch.load('my_transformer.pth'),strict=True)

    french=predict(net,english,tokenizer,max_output_len,device)
    print(french)

if __name__=="__main__":
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    for eng in engs:
        translate(eng)