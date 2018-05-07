
from util import DataManager, Encoder, Decoder
assert DataManager and Encoder and Decoder
import torch


BATCH_SIZE= 256
EPOCHS= 100
LATENT_DIM= 512
KL_DIVERGANCE_COEFFICIENT=0.5
OUTPUT_DIR= './data/reconstruction'

dm = DataManager()
train_shape=dm.get_data('train',['./data/train'],BATCH_SIZE, shuffle=True)
test_shape=dm.get_data('val',['./data/test'],BATCH_SIZE, shuffle=False)
assert(train_shape== test_shape)
data_shape=train_shape

encoder= Encoder(data_shape, LATENT_DIM).cuda()
decoder= Decoder(LATENT_DIM, data_shape).cuda()
optimizer= [torch.optim.Adam(encoder.parameters()), torch.optim.Adam(decoder.parameters())]

record=1
for epoch in range(1,EPOCHS+1):
    dm.train_vae('train', encoder, decoder, optimizer, epoch, print_every=5)
    record=dm.val_vae('val', encoder, decoder, optimizer, epoch, print_every=5,record=record, path=OUTPUT_DIR)
