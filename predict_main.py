import torch

from my_transformer.model import ModelArgs, Transformer
from my_transformer.tokenizer import load_data_fra
from my_transformer.train_predict import predict


def translate(english:list[str],data_path:str):

    device = torch.device(f'cuda:0')
    max_output_len=10

    # Get the same src_vocab and tgt_vocab as train. Otherwise, the model will not work.
    _, src_vocab, tgt_vocab = load_data_fra(data_path, batch_size=64,num_steps=10, num_examples=600)
    #I don't know why I still had to manually set dropout to 0.0 even after I set net.eval().
    args_enc = ModelArgs(vocab_size=len(src_vocab),dropout=0.0)
    args_dec = ModelArgs(vocab_size=len(tgt_vocab),dropout=0.0)

    net = Transformer(args_enc, args_dec)
    net.load_state_dict(torch.load('./non_code_files/my_transformer3.pth',weights_only=True), strict=True)
    net.to(device)

    french=[]
    for eng in english:
        french.append(predict(net,eng,src_vocab, tgt_vocab,max_output_len,device))
    return french

if __name__=="__main__":
    data_path = "./non_code_files/fra.txt"
    english = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .',
            'cheers !', 'go now .', 'i\'m a good guy .', 'i\'m a real man .']
    french=translate(english,data_path)
    print(french)