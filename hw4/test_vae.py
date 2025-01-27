
from util import DataManager, Encoder, Decoder
import numpy as np
import torch
import torch.nn.functional as F
import sys
assert DataManager and Encoder and Decoder and F and np and sys


torch.manual_seed(923)
BATCH_SIZE=  64
LATENT_DIM= 512
LABEL_ID= (7,8)
INPUT_DIR= './record'
DATA_DIR = sys.argv[1]
OUTPUT_DIR= sys.argv[2]

dm = DataManager(LATENT_DIM)
test_shape, test_label_dim=dm.get_data('test', i_path=['{}/test'.format(DATA_DIR)], c_path= ['{}/test.csv'.format(DATA_DIR)],class_range= LABEL_ID, mode= 'vae', batch_size= BATCH_SIZE, shuffle=False)

encoder= torch.load('encoder.pt')
decoder= torch.load('decoder.pt')
#print(encoder)
#print(decoder)
optimizer= [torch.optim.Adam(encoder.parameters()), torch.optim.Adam(decoder.parameters())]

###############################################################
#                       fig 1_2                               #
###############################################################
train_record= np.load('{}/vae_train_record.npy'.format(INPUT_DIR))
dm.plot_record(train_record, '{}/fig1_2.jpg'.format(OUTPUT_DIR),['Vae_train_reconstruction_loss','Vae_train_KL_loss'])

###############################################################
#                       fig 1_3 1_4                           #
###############################################################
dm.val_vae('test', encoder, decoder, optimizer,  print_every=5,path=OUTPUT_DIR, reconstruct_i_path = '{}/fig1_3.jpg'.format(OUTPUT_DIR),
                                            sample_i_path = '{}/fig1_4.jpg'.format(OUTPUT_DIR))
###############################################################
#                       fig 1_5                               #
###############################################################
dm.visualize_latent_space('test', encoder, '{}/fig1_5.jpg'.format(OUTPUT_DIR))
