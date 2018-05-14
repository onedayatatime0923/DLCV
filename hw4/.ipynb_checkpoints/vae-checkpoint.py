
from util import DataManager, Encoder, Decoder
import torch
import torch.nn.functional as F
assert DataManager and Encoder and Decoder and F


BATCH_SIZE=  64
EPOCHS= 40
LATENT_DIM= 512
HIDDEN_SIZE= 256
KL_DIVERGANCE_COEFFICIENT= torch.linspace(5E-4,1E-5,EPOCHS).cuda()
OUTPUT_DIR= './data/vae'

dm = DataManager(LATENT_DIM)
train_shape=dm.get_data('train',['./data/train'],BATCH_SIZE, shuffle=True)
test_shape=dm.get_data('val',['./data/test'],BATCH_SIZE, shuffle=False)
assert(train_shape== test_shape)
data_shape=train_shape

encoder= Encoder(data_shape, HIDDEN_SIZE, LATENT_DIM).cuda()
decoder= Decoder(LATENT_DIM, HIDDEN_SIZE, data_shape).cuda()
print(encoder)
print(decoder)
input()
optimizer= [torch.optim.Adam(encoder.parameters()), torch.optim.Adam(decoder.parameters())]

record=1
for epoch in range(1,EPOCHS+1):
    dm.train_vae('train', encoder, decoder, optimizer, epoch, KL_DIVERGANCE_COEFFICIENT[epoch-1], print_every=5)
    record=dm.val_vae('val', encoder, decoder, optimizer, epoch, print_every=5,record=0, path=OUTPUT_DIR)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')
