
from util import DataManager, Encoder, Generator ,Discriminator
import torch
import numpy as np
import sys
assert DataManager and Encoder and Generator and Discriminator


torch.manual_seed(1206)
BATCH_SIZE=  128
LATENT_DIM= 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
INPUT_DIR= './record'
OUTPUT_DIR= sys.argv[1]

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
#train_shape =dm.get_data('train',i_path=['./data/train','./data/test'], mode= 'gan', batch_size= BATCH_SIZE, shuffle=True)
#data_shape=train_shape

generator= torch.load('generator.pt')
discriminator= torch.load('discriminator.pt')
#print(generator)
#print(discriminator)

###############################################################
#                       fig 1_2                               #
###############################################################
train_record= np.load('{}/gan_train_record.npy'.format(INPUT_DIR))
dm.plot_record(train_record, '{}/fig2_2.jpg'.format(OUTPUT_DIR),['Gan_train_D_loss','Gan_train_G_loss'])

###############################################################
#                       fig 1_3                               #
###############################################################
dm.val_gan(generator, discriminator,  n=32, path=OUTPUT_DIR, sample_i_path= '{}/fig2_3.jpg'.format(OUTPUT_DIR))
