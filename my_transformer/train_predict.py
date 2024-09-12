import torch
from torch import nn

from my_transformer.tokenizer import Tokenizer
from my_transformer.utils import sequence_mask


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred.shape==(batch_size,num_steps,vocab_size)
    # label.shape==(batch_size,num_steps)
    # valid_len.shape==(batch_size)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train(net, data_iter, lr, num_epochs, tokenizer:Tokenizer, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    #net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    for epoch in range(num_epochs):
        train_loss,num_tokens=0,0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tokenizer.bos_id] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # teacher forcing
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  #
            train_loss=l.sum()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}: loss {train_loss/num_tokens:.3f}")

    print("End of training.")

def predict(net, src_sentence:str,tokenizer:Tokenizer, num_steps,device):
    net.eval()

    src_tokens = tokenizer.encode(src_sentence,bos=False,eos=True)
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # add batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_out = net.encoder(enc_X, enc_valid_len)
    # add batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tokenizer.bos_id], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y = net.decoder(dec_X, enc_out,enc_valid_len)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        if pred == tokenizer.eos_id:
            break
        output_seq.append(pred)
    return tokenizer.decode(output_seq)
