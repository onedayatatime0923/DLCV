
from util import DataManager


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''                load data                       '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=DataManager()
dm.load_image('data','image')
dm.pca_construct('image',240)
#print(dm.data['image'].shape)
#print(S.shape)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''                problem 2.1                     '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.plot(dm.pca.mean_,'pic/image_mean.png',title='mean face')
for i in range(3):
    dm.plot(dm.pca.components_[i],'pic/image_eigenvector{}.png'.format(str(i)),title='eigenface{}'.format(i))
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''                problem 2.2                     '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
x=dm.data['image'][0].reshape((1,-1))
for i in [3,50,100,239,399]:
    dm.pca_construct('image',i)
    y=dm.pca_reconstruct(dm.pca_transform(x)).reshape((1,-1))
    dm.plot(y,'pic/image_reconstruction_n={}'.format(i),title='n={}\nMSE={}'.format(i,((x-y)**2).mean() ))
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''                problem 2.3                     '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
for k in [1,3,5]:
    for n in [3,50,159]:
        print('k={}, n={}\t| validation accu={}'.format(k,n,dm.KNN(k,n,'val')))
print('k=1, n=159\t| prediction accu={}'.format(dm.KNN(1,159,'test')))
