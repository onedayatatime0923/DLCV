
from util import DataManager, Encoder, Generator ,Discriminator_Acgan
import torch
import numpy as np
assert DataManager and Encoder and Generator and Discriminator_Acgan


BATCH_SIZE=  128
EPOCHS= 100
LATENT_DIM= 128
LABEL_ID= (7,8)
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
INPUT_DIR= './record'
OUTPUT_DIR= './data/acgan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
data_size, label_dim=dm.get_data('train', i_path=['./data/train','./data/test'], c_path= ['./data/train.csv','./data/test.csv'],class_range=LABEL_ID,mode= 'gan', batch_size= BATCH_SIZE, shuffle=True)

generator= torch.load('generator_acgan.pt')
discriminator= torch.load('discriminator_acgan.pt')
optimizer= [generator.optimizer( lr=1E-4, betas= (0.5,0.999)),discriminator.optimizer( lr=1E-4, betas= (0.5,0.999))]
print(generator)
print(discriminator)

torch.save(generator,'generator_acgan.pt')
torch.save(discriminator,'discriminator_acgan.pt')
###############################################################
#                       fig 1_2                               #
###############################################################
train_record= np.load('{}/gan_train_record.npy'.format(INPUT_DIR))
dm.plot_record(train_record, '{}/fig3_2.jpg'.format(OUTPUT_DIR),['train_D_loss','train_G_loss'])

###############################################################
#                       fig 1_3                               #
###############################################################
dm.val_acgan(generator, discriminator,  n=10, path=OUTPUT_DIR, sample_i_path= '{}/fig3_3.jpg'.format(OUTPUT_DIR))
