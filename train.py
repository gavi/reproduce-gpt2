import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = open('input.txt').read()
chars = sorted(set(data))

atoi = {ch:i for i,ch in enumerate(chars)}
itoa = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [atoi[ch] for ch in s]
decode = lambda lst: ''.join([itoa[i] for i in lst])

print(encode('hello'))
print(decode(encode('hello')))

#lets create the train and validation groups
data_encoded = torch.tensor(encode(data))
split = int(len(data)*.9)
data_train = data_encoded[:split]
data_valid = data_encoded[split:]

print(len(data_encoded),len(data_train),len(data_valid))

vocab_size = len(chars)
batch_size = 4
context_size = 8

def get_batch(split):
    data = data_train if split=='train' else data_valid
    rands = torch.randint(0,len(data)-context_size-1,(batch_size,))
    x = torch.stack([data[i:i+context_size] for i in rands])
    y = torch.stack([data[i+1:i+context_size+1] for i in rands])
    return x,y

#lets build a simple Bigram Model

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self,x,targets=None):
        logits = self.embedding_table(x)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            source = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(source,targets)

        return logits,loss
    
    @torch.no_grad
    def generate(self,x,max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(x)
            #extract last item from the prediction

            logits = logits[:,-1,:]
            
            #now get softmax
            preds = F.softmax(logits, dim=-1) #B,1

            #lets pick one of the sample

            sample = torch.multinomial(preds, num_samples=1)
            #append it - autoregressive
            x = torch.cat((x,sample),dim=-1)
        return x

model = Bigram()
x,y = get_batch('train')

logits, loss = model(x,y)
print(loss)

input = torch.tensor([[0]]) #one batch with one token - new line character in this case
print(decode(model.generate(input, max_tokens=200)[0].tolist()))

#lets train 

optimizer = optim.AdamW(model.parameters(), lr = 1e-3)

for _ in range(10000):
    x, y = get_batch('train')
    logits, loss = model.forward(x,y)
    print(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(input, max_tokens=200)[0].tolist()))
