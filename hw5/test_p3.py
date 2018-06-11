

import torch
import argparse
from util import DataManager, MovieDataLoader

parser = argparse.ArgumentParser(description='DLCV HW5')
parser.add_argument('-v','--video_dir', dest='video_dir',required=True)
parser.add_argument('-o','--output', dest='output',required=True)
args = parser.parse_args()
dm= DataManager()

BATCH_SIZE =  8
INPUT_PATH = './model/rnn_movie.pt'
VIDEO_DIR = args.video_dir
TEST_DIR = args.output



val_data, test_path = dm.get_movie(VIDEO_DIR, None, save_path=None)

model= torch.load(INPUT_PATH)

val_dataloader= MovieDataLoader(val_data,None,batch_size= BATCH_SIZE, shuffle= False)

dm.write_movie(*dm.test_movie(model, val_dataloader, 0),TEST_DIR, test_path)
