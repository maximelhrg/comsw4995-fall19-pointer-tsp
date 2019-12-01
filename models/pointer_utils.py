import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from models.pointer_network import PointerNetwork


def adjusted_loss(outputs, y):
    first_term = F.nll_loss(outputs, y) + 1
    inp_size = y.size(1)
    
    M = torch.matmul(outputs, torch.transpose(outputs, 1, 2)) - torch.eye(20)
    N = torch.matmul(M, torch.transpose(M, 1, 2))
    second_term = torch.mean(torch.stack([torch.trace(m) / inp_size for m in torch.unbind(N)]))
    
    return first_term + second_term
    
    
# from pointer_network import PointerNetwork
def train_pointer(model, X, Y, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
    N = X.size(0)
    L = X.size(1)
    M = X.size(2)
    
    print('Training on {} batches of size {} for {} epochs'.format(N // batch_size, batch_size, n_epochs))
    print('\n')
    
    for epoch in range(n_epochs + 1):
        for i in range(0, N, batch_size):
            x = X[i: i + batch_size] # (bs, L, M)
            y = Y[i: i + batch_size] # (bs, L, M)

            probs = model(x) # (bs, L, M)
            outputs = F.softmax(probs, dim=2).permute(0, 2, 1) # (bs, L, M)
            
            loss = adjusted_loss(outputs, y)
            
            loss.backward(retain_graph = True)
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            test_pointer(model, X, Y)


def test_pointer(model, X, Y):
    model.eval()
    probs = model(X) # (bs, M, L)
    _v, indices = torch.max(probs, 2) # (bs, M)
    correct_count = sum([sum(ind.data == y.data).tolist() for ind, y in zip(indices, Y)])
    print('Acc: {:.2f}% ({}/{})'.format(correct_count / (X.size(0) * X.size(1)) * 100, correct_count, X.size(0) * X.size(1)))