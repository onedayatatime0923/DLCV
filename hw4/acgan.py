
from util import DataManager, Encoder, Generator ,Discriminator_Acgan
import torch
assert DataManager and Encoder and Generator and Discriminator_Acgan


BATCH_SIZE=  128
EPOCHS= 300
LATENT_DIM= 128
GENERATOR_HIDDEN_CHANNEL = 128
DISCRIMINATOR_HIDDEN_CHANNEL = 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
OUTPUT_DIR= './data/acgan'
TENSORBOARD_DIR= './runs/acgan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
dm.tb_setting(TENSORBOARD_DIR)
data_size, label_size=dm.get_data('train', i_path=['./data/train','./data/test'], c_path= ['./data/train.csv','./data/test.csv'],mode= 'acgan', batch_size= BATCH_SIZE, shuffle=True)

generator= Generator(LATENT_DIM+ label_size, GENERATOR_HIDDEN_CHANNEL, data_size[0]).cuda()
discriminator= Discriminator_Acgan( data_size[0], DISCRIMINATOR_HIDDEN_CHANNEL, label_size).cuda()
optimizer= [generator.optimizer(),discriminator.optimizer()]
print(generator)
print(discriminator)
#dm.tb_graph((generator,discriminator), LATENT_DIM)

for epoch in range(1,EPOCHS+1):
    dm.train_acgan('train', generator, discriminator, optimizer, epoch, print_every=5)
    dm.val_acgan(generator, discriminator, label=[0,1], epoch= epoch, n=10, path=OUTPUT_DIR)
torch.save(generator,'generator_acgan.pt')
torch.save(discriminator,'discriminator_acgan.pt')
