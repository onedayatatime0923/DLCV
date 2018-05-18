
from util import DataManager, Encoder, Decoder
import numpy as np
import torch
import torch.nn.functional as F
assert DataManager and Encoder and Decoder and F


BATCH_SIZE=  64
EPOCHS= 50
LATENT_DIM= 512
LABEL_ID= (7,8)
HIDDEN_SIZE= 256
KL_DIVERGANCE_COEFFICIENT= 8E-5
OUTPUT_DIR= './data/vae'
TENSORBOARD_DIR= './runs/vae'

dm = DataManager(LATENT_DIM)
dm.tb_setting(TENSORBOARD_DIR)
train_shape, train_label_dim=dm.get_data('train', i_path=['./data/train'], c_path= ['./data/train.csv'],class_range= LABEL_ID, mode= 'vae', batch_size= BATCH_SIZE, shuffle=True)
test_shape, test_label_dim=dm.get_data('val', i_path=['./data/test'], c_path= ['./data/test.csv'],class_range= LABEL_ID, mode= 'vae', batch_size= BATCH_SIZE, shuffle=False)
assert(train_shape== test_shape)
assert(train_label_dim == test_label_dim == 1)
data_shape=train_shape

encoder= Encoder(data_shape, HIDDEN_SIZE, LATENT_DIM).cuda()
decoder= Decoder(LATENT_DIM, HIDDEN_SIZE, data_shape).cuda()
print(encoder)
print(decoder)
optimizer= [torch.optim.Adam(encoder.parameters()), torch.optim.Adam(decoder.parameters())]

train_record=[]
test_record=[]
for epoch in range(1,EPOCHS+1):
    train_record.append(dm.train_vae('train', encoder, decoder, optimizer, epoch, KL_DIVERGANCE_COEFFICIENT, print_every=5))
    test_record.append(dm.val_vae('val', encoder, decoder, optimizer, epoch, print_every=5, path=OUTPUT_DIR))
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')
#print(np.array(train_record))
#print(np.array(test_record))
np.save('record/vae_train_record.npy', np.array(train_record))
np.save('record/vae_test_record.npy', np.array(test_record))
