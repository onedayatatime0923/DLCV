
from util import DataManager, Encoder, Generator
import torch
assert DataManager and Encoder and Generator and torch


BATCH_SIZE= 256
EPOCHS= 100
LATENT_DIM= 512
DISCRIMINATOR_UPDATE_NUM= 1
GENERATOR_UPDATE_NUM= 1
OUTPUT_DIR= './data/gan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
train_shape=dm.get_data('train',['./data/train','./data/test'],BATCH_SIZE, shuffle=True)
data_shape=train_shape

'''
encoder= Encoder(data_shape, LATENT_DIM).cuda()
decoder= Generator(LATENT_DIM, data_shape).cuda()
optimizer= [torch.optim.Adam(encoder.parameters()), torch.optim.Adam(decoder.parameters())]

record=1
for epoch in range(1,EPOCHS+1):
    dm.train('train', encoder, decoder, optimizer, epoch, print_every=5)
    record=dm.val('val', encoder, decoder, optimizer, epoch, print_every=5,record=record, path=OUTPUT_DIR)
    '''
