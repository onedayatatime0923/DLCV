
from util import Datamanager
import numpy as np
assert np

dm=Datamanager()
color=[[0,0,0],[255,0,0],[0,255,0],[0,0,255],
       [255,255,0],[0,255,255],[255,0,255],[255,255,255],
       [63,191,31],[127,127,127]]

###############################################################
#                    problem 2.1                              #
###############################################################
'''
for i in ['zebra','mountain']:
    dm.read_image('image','{}.jpg'.format(i),mode='rgb')
    dm.kmeans_cluster(dm.image.reshape((-1,3)))
    image=np.array([color[dm.kmeans.labels_[i]] for i in range(len(dm.kmeans.labels_))]).reshape(dm.image.shape[0],dm.image.shape[1],-1)
    dm.plot(image,'pic/{}_rgb.png'.format(i))

    dm.read_image('image','{}.jpg'.format(i),mode='lab')
    dm.kmeans_cluster(dm.image.reshape((-1,3)))
    image=np.array([color[dm.kmeans.labels_[i]] for i in range(len(dm.kmeans.labels_))]).reshape(dm.image.shape[0],dm.image.shape[1],-1)
    dm.plot(image,'pic/{}_lab.png'.format(i))

'''
###############################################################
#                    problem 2.2                              #
###############################################################
