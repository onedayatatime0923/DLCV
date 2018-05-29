
from util import DataManager


dm= DataManager()
dm.get_data('./data/TrimmedVideos/video/train', './data/TrimmedVideos/label/gt_train.csv', save_path=['./data/trainx.npy','./data/trainy.npy'])
