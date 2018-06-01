

import torch
from torch import nn
from torch.autograd import Variable
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden= self.initHidden(10)

        self.classifier = nn.Sequential(
                nn.Linear( 64,64))
    def forward(self, x, i):
        #print(x.size())
        #print(i)
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        #print(packed_data.data[0])
        #print(i)
        #print(x.size())
        #print(packed_data.data.size())
        z = self.features(packed_data.data)
        z = z.view(z.size(0), -1)
        z = self.dimention_reduction(z)
        #print(packed_data.batch_sizes)
        #print(z.data[0])
        packed_data=nn.utils.rnn.PackedSequence(z, packed_data.batch_sizes)
        packed_data, _=self.rnn(packed_data, self.hidden_layer(len(x)))
        #print(packed_data.batch_sizes)
        #print(packed_data.data[:3])
        #input()
        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)
        #print(z[0].size())
        z = torch.sum(z[0],1)/ i.unsqueeze(1).repeat(1,z[0].size(2)).float()
        #print(z.size())
        #print(sort_i)
        #input()
        z = self.classifier(z)
        #print(torch.index_select(sort_i, 0, sort_index_reverse))
        #input()
        
        return z
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(1,1, hidden_size),requires_grad=True).cuda()


for i in list(Model().parameters()):
    print(i.size())
