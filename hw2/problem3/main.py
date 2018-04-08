
from util import Datamanager
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
assert np and os and plt and Axes3D

dm=Datamanager()
color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

###############################################################
#                    problem 3.1                              #
###############################################################
'''

dm.read_image('image','train-10/Suburb/image_0029.jpg',mode='rgb')
dm.surf_detect(dm.data['image'])
dm.surf_plot(dm.data['image'],'pic/3_1_surf029.png')

'''
###############################################################
#                    problem 3.2                              #
###############################################################
'''
feature=[]
for d in os.listdir('train-10'):
    for f in os.listdir('train-10/{}'.format(d)):
        dm.read_image('image','train-10/{}/{}'.format(d,f),mode='rgb')
        feature.append(dm.surf_detect(dm.data['image']))
feature=np.concatenate(feature,0)

dm.kmeans_construct(feature)
y=dm.kmeans.labels_
dm.pca_construct(feature,3)
x=dm.pca_transform(feature)
#print(x,y)
#print(x.shape,y.shape)
y_center=np.arange(6)
x_center=dm.pca_transform(dm.kmeans.cluster_centers_[:6])
#print(x_center,y_center)
#rint(x_center.shape,y_center.shape)


fig = plt.figure()
ax = Axes3D(fig)
for i in range(len(x)):
    if y[i]<6:
        ax.scatter(x[i,0],x[i,1],x[i,2],s=3,c=color[y[i]])
for i in range(len(x_center)):
    ax.scatter(x_center[i,0],x_center[i,1],x_center[i,2],s=10,c=color[y_center[i]])
plt.savefig('pic/3_2pca_cluster.png')

'''
###############################################################
#                    problem 3.3                              #
###############################################################
'''
feature=[]
for d in os.listdir('train-10'):
    for f in os.listdir('train-10/{}'.format(d)):
        dm.read_image('image','train-10/{}/{}'.format(d,f),mode='rgb')
        feature.append(dm.surf_detect(dm.data['image']))
feature=np.concatenate(feature,0)
dm.kmeans_construct(feature)

dm.read_image('image','train-10/Forest/image_0003.jpg',mode='rgb')
mode=[('soft','max'),('soft','sum'),('hard','sum')]
for  i in mode:
    assign_=dm.kmeans_assign(dm.surf_detect(dm.data['image']),mode=i[0])
    #print(assign_.shape)
    emb=dm.pooling(assign_,mode=i[1])
    #print(emb.shape)
    dm.plot_bar(emb,'pic/3_3_histogram_{}_{}.png'.format(i[0],i[1]))
'''
###############################################################
#                    problem 3.3                              #
###############################################################
'''
'''
train_dir='train-10'
test_dir='test-100'
train_dir_list=os.listdir(train_dir)
test_dir_list=os.listdir(test_dir)
train_x=[]
train_y=[]
for i in range(len(train_dir_list)):
    for f in os.listdir('{}/{}'.format(train_dir,train_dir_list[i])):
        dm.read_image('image','{}/{}/{}'.format(train_dir,train_dir_list[i],f),mode='rgb')
        train_x.append(dm.surf_detect(dm.data['image']))
        train_y.append(i)
train_x=np.array(train_x)
train_y=np.array(train_y)
feature=np.concatenate(train_x,0)
dm.kmeans_construct(feature)

test_x=[]
test_y=[]
for i in range(len(test_dir_list)):
    for f in os.listdir('{}/{}'.format(test_dir,test_dir_list[i])):
        dm.read_image('image','{}/{}/{}'.format(test_dir,test_dir_list[i],f),mode='rgb')
        test_x.append(dm.surf_detect(dm.data['image']))
        test_y.append(i)

mode=[('soft','max'),('hard','sum'),('soft','sum')]
for m in mode:
    train_x_emb=np.array([dm.embedding(i,m) for i in train_x])
    dm.KNN_construct(train_x_emb,train_y,len(train_dir))

    pred=np.array([dm.KNN_predict([dm.embedding(i,m)])[0] for i in test_x])
    correct=(test_y==pred).sum()
    print('Mode {} {} | Accuracy: {}%'.format(m[0],m[1],100.*correct/len(pred)))
'''
'''
