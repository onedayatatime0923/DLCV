
from util import DataManager, Encoder, Generator ,Discriminator
assert DataManager and Encoder and Generator and Discriminator


BATCH_SIZE=  128
EPOCHS= 300
LATENT_DIM= 128
GENERATOR_HIDDEN_CHANNEL = 128
DISCRIMINATOR_HIDDEN_CHANNEL = 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
OUTPUT_DIR= './data/gan'
TENSORBOARD_DIR= './runs/gan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
dm.tb_setting(TENSORBOARD_DIR)
train_shape=dm.get_data('train',['./data/train','./data/test'],mode= 'gan', batch_size= BATCH_SIZE, shuffle=True)
data_shape=train_shape

generator= Generator(LATENT_DIM, GENERATOR_HIDDEN_CHANNEL, data_shape[0]).cuda()
discriminator= Discriminator( data_shape[0], DISCRIMINATOR_HIDDEN_CHANNEL).cuda()
optimizer= [generator.optimizer(),discriminator.optimizer()]
print(generator)
print(discriminator)
dm.tb_graph((generator,discriminator), LATENT_DIM)

record=1
for epoch in range(1,EPOCHS+1):
    dm.train_gan('train', generator, discriminator, optimizer, epoch, print_every=5)
    dm.val_gan(generator, discriminator, epoch, n=32, path=OUTPUT_DIR)
