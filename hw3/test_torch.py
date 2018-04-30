

from util import Datamanager
import torch

INPUT_SIZE= (512,512,3)
TEST_SIZE= 257
BATCH_SIZE= 16

dm = Datamanager()


dm.get_data('test', 'test',  './data/validation', TEST_SIZE, './data/test_x.npy')
testloader= dm.dataloader(dm.data['test'],BATCH_SIZE,shuffle=False)

#model=dm.fcn32('./data/vgg16_bn.pth').cuda()
#model.load_state_dict(torch.load('./data/model.pt'))
model=torch.load('./data/model.pt')

im=dm.test(model, testloader)
dm.write(im,'./data/test/')

