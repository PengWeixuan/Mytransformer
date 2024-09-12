import torch

from my_transformer.model import ModelArgs, Transformer
from my_transformer.tokenizer import load_data_fra
from my_transformer.train_predict import train

def trainMyTransformer(data_path:str):
    torch.manual_seed(1)
    device = torch.device(f'cuda:0')

    batch_size = 64
    max_seq_len = 10
    lr = 0.005
    num_epochs = 200
    # The English-French pairs in the first "num_examples" line of fra.txt
    # file are used as training samples.
    num_examples=600

    train_iter, src_vocab, tgt_vocab = load_data_fra(data_path,batch_size=batch_size,
                                                     num_steps=max_seq_len,num_examples=num_examples)
    args_enc = ModelArgs(vocab_size=len(src_vocab))
    args_dec = ModelArgs(vocab_size=len(tgt_vocab))
    net=Transformer(args_enc,args_dec)
    # Uncomment this if you want train model based
    #net.load_state_dict(torch.load('./non_code_files/my_transformer.pth', weights_only=True), strict=True)

    train(net,train_iter,lr,num_epochs,tgt_vocab,device)
    torch.save(net.state_dict(), './non_code_files/my_transformer.pth')
    print("Weights saved.")

if __name__=="__main__":
    # Download data set from https://www.manythings.org/anki/fra-eng.zip
    data_path="./non_code_files/fra.txt"
    trainMyTransformer(data_path)
