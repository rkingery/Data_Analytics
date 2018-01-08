import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.signal import medfilt, wiener
from scipy.stats import norm

np.random.seed(123)

DATA_DIR = '/Users/ryankingery/da-project-files/Raw_Data/Second_Set/'

df = pd.read_csv(DATA_DIR+'psychometric_info.csv')
#df['user_id'] = df['user_id'].astype('category').cat.codes

new = df.drop(['employee_name','user_id'],axis=1)
X = new.values


for i in range(5):
    plt.hist(X[:,i])
    plt.xlabel(str(new.columns[i]))
    plt.show()
    
plt.boxplot(X)
plt.show()

outliers = []
for j in range(5):
    q1 = np.percentile(X[:,j],25)
    q2 = np.percentile(X[:,j],75)
    IQR = 1.5*(q2-q1)
    outliers += [ df[np.logical_or(X[:,j]<(q1-IQR), X[:,j]>(q2+IQR))] ]
print 'boxplot outliers = '+str(outliers)


outliers = []
comps = [2,2,2,2,1]
for j in range(5):
    x = X[:,j].reshape(-1,1)
    model = BayesianGaussianMixture(n_components=comps[j], covariance_type='full')
    model.fit(x)
    both = np.column_stack([x,model.predict(x)])
    



# attempts to cluster the whole feature space...may or may not be useful
#X_filt = medfilt(X)
num_clusters = np.arange(1,20+1)
scores = []
for num in num_clusters:
    #model = KMeans(n_clusters=num,max_iter=500,n_init=20)
    #model = GaussianMixture(n_components=num, covariance_type='full')
    model = BayesianGaussianMixture(n_components=num, covariance_type='full')
    model.fit(X)
    scores += [-model.score(X)]
    
plt.plot(np.arange(1,20+1),scores)
plt.xlim(0,20)
plt.xticks(np.arange(1,20+1))
plt.xlabel('number of clusters')
plt.ylabel('loss')
plt.show()

# 4,6,10 clusters?
#model = KMeans(n_clusters=4,max_iter=500,n_init=20)
model = BayesianGaussianMixture(n_components=10, covariance_type='full')
model.fit(X)
plt.hist(model.predict(X))
plt.show()
#print model.cluster_centers_


Z = linkage(X,method='average')
plt.figure()
dn = dendrogram(Z,p=50,truncate_mode='lastp',show_contracted=True,show_leaf_counts=False)
c, d = cophenet(Z,pdist(X))
print c
#
#fig = plt.figure()
#idx = 1
#for i in range(5):
#    for j in range(5):
#        ax = fig.add_subplot(5,5,idx)
#        ax.scatter(X[:,i],X[:,j])
#        idx+=1
#plt.show()

