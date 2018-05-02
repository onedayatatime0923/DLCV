
from util import Datamanager
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model
assert load_model

INPUT_SIZE= (512,512,3)
TRAIN_SIZE= 2313
VAL_SIZE= 257
EPOCHS= 100
BATCH_SIZE=  8
DROPOUT=0.5

dm = Datamanager()

dm.get_data('train', 'train',  './data/train', './data/train_x.npy','./data/train_y.npy')
dm.get_data('train', 'val',  './data/validation','./data/val_x.npy','./data/val_y.npy')
print('train_x shape: {}'.format(dm.data['train'][0].shape))
print('train_y shape: {}'.format(dm.data['train'][1].shape))
print('val_x shape: {}'.format(dm.data['val'][0].shape))
print('val_y shape: {}'.format(dm.data['val'][1].shape))

trainloader= dm.generate(dm.data['train'],BATCH_SIZE,shuffle=True)
valloader= dm.generate(dm.data['val'],BATCH_SIZE,shuffle=True)

model=dm.model(INPUT_SIZE,'./data/vgg16_weights_tf_dim_ordering_tf_kernels.h5',DROPOUT)
#model.save_weights('./data/model.h5')
model.summary()

############################################################
#                  setting checkpoint                      #
############################################################
#filepath="./data/model.h5"
filepath="./model/model_fcn32_{epoch:02d}.hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
callbacks_list = [checkpoint1,checkpoint2]

############################################################
#                  train                                   #
############################################################

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(trainloader, steps_per_epoch=(TRAIN_SIZE// BATCH_SIZE) +1, validation_data= valloader,validation_steps=(VAL_SIZE// BATCH_SIZE) +1, epochs=EPOCHS,callbacks=callbacks_list)

