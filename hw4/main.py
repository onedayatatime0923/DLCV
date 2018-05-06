
from util import DataManager, Encoder, Decoder
assert DataManager and Encoder and Decoder
import torch


BATCH_SIZE= 128
EPOCHS= 100
LATENT_DIM= 512

dm = DataManager()
train_shape=dm.get_data('train','./data/train',BATCH_SIZE, shuffle=True)
test_shape=dm.get_data('val','./data/test',BATCH_SIZE, shuffle=True)
assert(train_shape== test_shape)
data_shape=train_shape

encoder= Encoder(data_shape, LATENT_DIM)
decoder= Decoder(LATENT_DIM, data_shape)
optimizer= [torch.optim.Adam(encoder.parameters()), torch.optim.Adam(decoder.parameters())]

for epoch in range(1,EPOCHS+1):
    dm.train('train', encoder, decoder, optimizer, epoch, print_every=5)
    dm.val('train', encoder, decoder, optimizer, epoch, print_every=5)
