
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from util import DataManager
assert os and Image
dm= DataManager()

color=[(0,255,255),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(255,255,255),(0,0,0) ,(0,127,255),(127,255,0),(127,0,255),(200,73,100)]

predict_file= './test/OP01-R03-BaconAndEggs.txt'
ground_file= './data/FullLengthVideos/labels/valid/OP01-R03-BaconAndEggs.txt'
image_path ='./data/FullLengthVideos/videos/valid/OP01-R03-BaconAndEggs/'

image_list= os.listdir(image_path)
image_list.sort()
image_list=[ image_list[i] for i in range(0,300,50)]

images= []
for i in image_list:
    images.append(np.array(Image.open('{}/{}'.format(image_path,i)).resize((50,50))))
images= np.concatenate(images,1)

with open(predict_file, 'r') as f:
    predict= np.array([[color[int(i.strip())] for i in f.readlines()]])
with open(ground_file, 'r') as f:
    ground= np.array([[color[int(i.strip())] for i in f.readlines()]])

predict= np.tile(predict[:,:300,:],(50,1,1))
ground= np.tile(ground[:,:300,:],(50,1,1))

output= np.concatenate((ground, images, predict),0)

plt.figure()
plt.xticks(())
plt.yticks(())
plt.imshow(output)
plt.savefig('video_visualization.png')

color_pic= np.tile(np.array([[i for i in color for j in range(20)]]),(5,1,1))
#print(color_pic)
#print(color_pic.shape)


plt.figure()
plt.xticks(())
plt.yticks(())
plt.imshow(color_pic)
plt.savefig('color.png')
