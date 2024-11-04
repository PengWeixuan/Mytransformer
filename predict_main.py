import torch

from my_transformer.model import ModelArgs, Transformer
from my_transformer.tokenizer import Tokenizer
from my_transformer.train_predict import predict

def translate(English:list[str], tokenizer_path:str):
    device = torch.device(f'cuda:0')
    max_output_len=10

    tokenizer = Tokenizer(tokenizer_path)
    args = ModelArgs(vocab_size=tokenizer.n_words,dropout=0.0)
    net = Transformer(args)
    net.load_state_dict(torch.load('./non_code_files/my_transformer_4.pth',weights_only=True), strict=True)
    net.to(device)

    French=[]
    for eng in English:
        French.append(predict(net,eng,tokenizer,max_output_len,device))
    return French

if __name__=="__main__":
    tokenizer_path = "./non_code_files/tokenizer.model"
    English = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .',
            'cheers !', 'go now .', 'i\'m a good guy .', 'i\'m a real man .']
    french=translate(English,tokenizer_path)
    print(french)