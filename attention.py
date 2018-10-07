import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict

class global_attention(nn.Module):
    def __init__(self, hidden_size, activation=None):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)

        # self.gate = nn.Linear(2*hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, x, hiddens):
        gamma_h = self.linear_in(x).unsqueeze(2)    # batch * size * 1
        if self.activation == "tanh":
            gamma_h = self.tanh(gamma_h)
        #
        weight_c = torch.bmm(self.context, gamma_h).squeeze(2)   # batch * time
        weight_c = self.softmax(weight_c)   # batch * time
        c_t = torch.bmm(weight_c.unsqueeze(1), self.context).squeeze(1) # batch * size
        
        #
        weight_h = torch.bmm(hiddens, gamma_h).squeeze(2)
        weight_h = self.softmax(weight_h)
        c_h = torch.bmm(weight_h.unsqueeze(1), hiddens).squeeze(1)


        '''
        # gate
        gate = self.sigmoid(self.gate(torch.cat([c_t, c_h], 1)))
        c = gate * c_t + (1-gate) * c_h
        '''
        
        # attention
        t_h = torch.cat([c_t.unsqueeze(2), c_h.unsqueeze(2)], 2)
        weigh = torch.matmul(x.unsqueeze(1), t_h)
        weigh = self.softmax(weigh).transpose(1, 2)
        c = torch.matmul(t_h, weigh).squeeze(2)


        output = self.tanh(self.linear_out(torch.cat([x, c], 1)))
        
        return c_t, output, weight_c

