# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class PointerNetwork(nn.Module):
    """
    Define the Pointer Network used to learn TSP from heatmap given by the ConvNet.
    """
    def __init__(self, inp_size, weight_size, answer_seq_len, hidden_size=512, is_GRU=True):
        super(PointerNetwork, self).__init__()

        input_size = inp_size[0]
        emb_size = inp_size[1]
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.emb_size = emb_size
        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.is_GRU = is_GRU

        if is_GRU:
            self.enc = nn.GRU(emb_size, hidden_size, batch_first=True)
            self.dec = nn.GRUCell(emb_size, hidden_size) # GRUCell's input is always batch first
        else:
            self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
            self.dec = nn.LSTMCell(emb_size, hidden_size) # LSTMCell's input is always batch first

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False) # scaling sum of enc and dec by v.T

        
    def forward(self, input):
        batch_size = input.size(0)

        # Encoding
        encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = encoder_states.transpose(1, 0) # (L, bs, H)

        # Decoding states initialization
        decoder_input = to_var(torch.zeros(batch_size, self.emb_size)) # (bs, embd_size)
        hidden = to_var(torch.zeros([batch_size, self.hidden_size]))   # (bs, h)
        cell_state = encoder_states[-1]                                # (bs, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden) # (bs, h), (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)

            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)          # (L, bs, W)
            blend2 = self.W2(hidden)                  # (bs, W)
            blend_sum = F.tanh(blend1 + blend2)    # (L, bs, W)
            out = self.vt(blend_sum).squeeze()        # (L, bs)
            out = F.log_softmax(out.transpose(0, 1).contiguous(), -1) # (bs, L)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           # (bs, M, L)

        return probs
    

    
## Trying with a simple dense network
    
class DenseNTM(nn.Module):
    def __init__(self, inp_size):
        super().__init__()
        
        self.inp_size = inp_size
        self.size = inp_size[0] * inp_size[1]
        self.fc1 = nn.Linear(self.size, self.size)
        
    def forward(self, x):

        # x: (bs, L, M)
        x = x.view(-1, self.size) # (bs, L*M)
        x = F.tanh(self.fc1(x)) # (bs, L*M)
        x = x.reshape(-1, self.inp_size[0], self.inp_size[1]) # (bs, L, M)
        y = F.softmax(x) # (bs, L, M)
        
        return y