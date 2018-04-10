
from util import Datamanager
import numpy as np
assert np

dm=Datamanager()
color=[[0,0,0],[255,0,0],[0,255,0],[0,0,255],
       [255,255,0],[0,255,255],[255,0,255],[255,255,255],
       [63,191,31],[127,127,127]]

###############################################################
#                    problem 2.a                              #
###############################################################
'''
'''
for i in ['zebra','mountain']:
    dm.read_image('image','{}.jpg'.format(i),mode='rgb')
    dm.kmeans('kmeans',dm.data['image'].reshape((-1,3)))
    image=np.array([color[dm.data['kmeans'].labels_[i]] for i in range(len(dm.data['kmeans'].labels_))]).reshape((dm.data['image'].shape[0],dm.data['image'].shape[1],-1))
    dm.plot(image,'Cluster Method: RBG','pic/{}_rgb.png'.format(i))

    dm.read_image('image','{}.jpg'.format(i),mode='lab')
    dm.kmeans('kmeans',dm.data['image'].reshape((-1,3)))
    image=np.array([color[dm.data['kmeans'].labels_[i]] for i in range(len(dm.data['kmeans'].labels_))]).reshape((dm.data['image'].shape[0],dm.data['image'].shape[1],-1))
    dm.plot(image,'Cluster Method: LAB','pic/{}_lab.png'.format(i))

###############################################################
#                    problem 2.b                              #
###############################################################
for j in ['zebra','mountain']:
    dm.read_image('image','{}.jpg'.format(j),mode='L')
    dm.read_mat('filter','filterBank.mat')
    feature=[]
    for i in range(dm.data['filter'].shape[2]):
        im=dm.correlate(dm.data['image'],dm.data['filter'][:,:,i])
        feature.append(im)
    feature=np.array(feature).reshape((dm.data['filter'].shape[2],-1)).T
    dm.kmeans('kmeans',feature,n=6)
    image=np.array([color[dm.data['kmeans'].labels_[i]] for i in range(len(dm.data['kmeans'].labels_))]).reshape((dm.data['image'].shape[0],dm.data['image'].shape[1],-1))
    dm.plot(image,'Cluster Method: Filter+Kmeans','pic/{}_texture.png'.format(j))

    dm.read_image('image_lab','{}.jpg'.format(j),mode='lab')
    image=dm.data['image_lab'].reshape(-1,3)
    feature=np.concatenate((feature,image),1)
    dm.kmeans('kmeans',feature,n=6)
    image=np.array([color[dm.data['kmeans'].labels_[i]] for i in range(len(dm.data['kmeans'].labels_))]).reshape((dm.data['image'].shape[0],dm.data['image'].shape[1],-1))
    dm.plot(image,'Cluster Method: Filter+Kmeans+LAB','pic/{}_texture_lab.png'.format(j))
'''
'''
