import torch

from my_transformer.model import ModelArgs, Transformer
from my_transformer.tokenizer import load_data_fra
from my_transformer.train_predict import train

def trainMyTransformer(tokenizer_path,data_path):
    torch.manual_seed(1)
    device = torch.device(f'cuda:0')

    batch_size = 64
    num_steps = 10
    lr = 0.005
    num_epochs = 200

    train_iter, tokenizer = load_data_fra(data_path,tokenizer_path,batch_size=batch_size, num_steps=num_steps)
    args = ModelArgs(vocab_size=tokenizer.n_words)

    net=Transformer(args)

    train(net,train_iter,lr,num_epochs,tokenizer,device)
    torch.save(net.state_dict(), 'my_transformer.pth')
    print("Weights saved.")

if __name__=="__main__":
    pass
    tokenizer_path = "C:\\Users\\39936\\Downloads\\tokenizer.model"
    data_path="../non_code_files/fra.txt"
    trainMyTransformer(tokenizer_path,data_path)
