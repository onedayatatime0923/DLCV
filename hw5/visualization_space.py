
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import DataManager, ImageDataLoader
assert  ImageDataLoader

dm= DataManager()
x= np.load('./dataset/valx.npy')
y= np.load('./dataset/valy.npy')

################################################################
#                      cnn feature                             #
################################################################
OUTPUT_PATH = './cnn_visualization.png'

feature_x=[]
for i in x:
    feature_x.append(np.mean(i,0))
feature_x= np.array(feature_x)
#print(feature_x.shape)
#print(y.shape)

dm.visualize_latent_space(feature_x, y, OUTPUT_PATH)

################################################################
#                      rnn feature                             #
################################################################

INPUT_PATH = './model/rnn_classifier.pt'
OUTPUT_PATH = './rnn_visualization.png'
class Rnn_Classifier(nn.Module):
    def __init__(self, path):#input_dim, hidden_dim, layer_n, label_dim, dropout ):
        super(Rnn_Classifier, self).__init__()
        self.layer_n = torch.load(path).layer_n
        self.hidden= torch.load(path).hidden

        self.rnn= torch.load(path).rnn
    def forward(self, x, i):
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)

        packed_data, hidden=self.rnn(packed_data, self.hidden_layer(len(x)))

        z = hidden.permute(1,0,2).contiguous().view(hidden.size(1),-1)
        
        return z
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda(),requires_grad=True)
    def save(self, path):
        torch.save(self,path)

def train_rnn(model, dataloader):
    model.train()
    
    feature_data=[]
    feature_label=[]
    for b, (x, i, y, _) in enumerate(dataloader):
        x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
        output= model(x,i)
        #print(output.size())
        #print(y.size())
        #input()
        feature_data.append(output.detach().data)
        feature_label.append(y.detach().data)
    feature_data= torch.cat(feature_data,0)
    feature_label= torch.cat(feature_label,0)
    return feature_data, feature_label

train_dataloader= ImageDataLoader(x, y,batch_size= 32, shuffle= True)
model= Rnn_Classifier(INPUT_PATH)

feature_data, feature_label =train_rnn( model, train_dataloader)
dm.visualize_latent_space(feature_data, feature_label, OUTPUT_PATH)


