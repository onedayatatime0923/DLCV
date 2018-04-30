

from util import Datamanager
import numpy as np
from keras.models import load_model
import argparse

BATCH_SIZE=  8

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='validation data input directory', required=True, type=str)
parser.add_argument('-o', '--output', help='validation data output directory', required=True, type=str)
parser.add_argument('-m', '--model', help='validation data output directory', default='./data/model_fcn16.h5', type=str)
args = parser.parse_args()

dm = Datamanager()

dm.get_data('test', 'test',  args.input )

model=load_model(args.model)

im=model.predict(dm.data['test'], batch_size=BATCH_SIZE, verbose=1)
output=np.argmax(im,3)
dm.write(output,args.output)

