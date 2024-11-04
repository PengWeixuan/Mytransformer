import torch

from my_transformer.model import ModelArgs, Transformer
from my_transformer.tokenizer import load_data_fra
from my_transformer.train_predict import train

def trainMyTransformer(data_path:str,tokenizer_path:str):
    torch.manual_seed(1)
    device = torch.device(f'cuda:0')

    batch_size = 64
    lr = 0.001
    num_epochs = 100
    # The English-French pairs in the first "num_examples" lines of fra.txt
    # file are used as training samples. Each sentence is no longer than
    # "max_seq_len" tokens.
    num_examples=600 # There are 167130 lines in total in fra.txt.
    max_seq_len = 10

    train_iter, tokenizer = load_data_fra(data_path,tokenizer_path,batch_size=batch_size,
                                          num_steps=max_seq_len,num_examples=num_examples)
    args = ModelArgs(vocab_size=tokenizer.n_words)

    net=Transformer(args)
    # Uncomment this if you want train model based
    net.load_state_dict(torch.load('./non_code_files/my_transformer_4.pth', weights_only=True), strict=True)

    train(net,train_iter,lr,num_epochs,tokenizer,device)
    torch.save(net.state_dict(), './non_code_files/my_transformer_5.pth')
    print("Weights saved.")

if __name__=="__main__":
    tokenizer_path = "./non_code_files/tokenizer.model"
    # Download data set from https://www.manythings.org/anki/fra-eng.zip
    data_path="./non_code_files/fra.txt"
    trainMyTransformer(data_path,tokenizer_path)
