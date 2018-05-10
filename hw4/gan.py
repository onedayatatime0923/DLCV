
from util import DataManager, Encoder, Generator ,Discriminator
import torch
assert DataManager and Encoder and Generator and Discriminator


BATCH_SIZE= 128
EPOCHS= 100
LATENT_DIM= 512
GENERATOR_HIDDEN_SIZE = 512
DISCRIMINATOR_HIDDEN_SIZE = 512
DISCRIMINATOR_UPDATE_NUM= 1
GENERATOR_UPDATE_NUM= 1
OUTPUT_DIR= './data/gan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
train_shape=dm.get_data('train',['./data/train','./data/test'],BATCH_SIZE, shuffle=True)
data_shape=train_shape

generator= Generator(LATENT_DIM, GENERATOR_HIDDEN_SIZE, data_shape).cuda()
discriminator= Discriminator(data_shape,DISCRIMINATOR_HIDDEN_SIZE, 1).cuda()
print(generator)
print(discriminator)
optimizer= [torch.optim.Adam(generator.parameters()), torch.optim.Adam(discriminator.parameters())]

record=1
for epoch in range(1,EPOCHS+1):
    dm.train_gan('train', generator, discriminator, optimizer, epoch, print_every=5)
    dm.val_gan(generator, discriminator, n=5, path=OUTPUT_DIR)
