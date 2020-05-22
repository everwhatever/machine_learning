import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data=make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=45)

kmeans=KMeans(n_clusters=4)
kmeans.fit(data[0])

fig,axes=plt.subplots(1,2)
axes[0].scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
axes[0].set_title('Original')
axes[1].scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
axes[1].set_title('KMeans')



#plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()
