

import argparse
import torch
from util import DataManager, ImageDataLoader

parser = argparse.ArgumentParser(description='DLCV HW5')
parser.add_argument('-v','--video_dir', dest='video_dir',required=True)
parser.add_argument('-t','--tag', dest='tag',required=True)
parser.add_argument('-o','--output', dest='output',required=True)
args = parser.parse_args()
dm= DataManager()

BATCH_SIZE = 128
INPUT_PATH = './model/rnn_classifier.pt'
TEST_PATH = args.output
VIDEO_DIR = args.video_dir
TAG = args.tag

valx, valy= dm.get_data(VIDEO_DIR, TAG, save_path=None)

model= torch.load(INPUT_PATH)

val_dataloader= ImageDataLoader(valx,valy,batch_size= BATCH_SIZE, shuffle= False)

dm.write(dm.test_rnn(model, val_dataloader, 0),TEST_PATH)
