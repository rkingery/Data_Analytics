import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import decomposition
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.signal import medfilt, wiener
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
np.random.seed(42)

DATA_DIR = '/Users/ryankingery/da-project-files/Raw_Data/Second_Set/'

df = pd.read_csv(DATA_DIR+'logon_info.csv')
#df['activity'] = df['activity'].replace({'Logon':1,'Logoff':0})
#df['date'] = pd.to_datetime(df['date'])
#df['user'] = df['user'].astype('category').cat.codes
#df['pc'] = df['pc'].astype('category').cat.codes

new = df.drop(['id','date'],axis=1)
new_dum = pd.get_dummies(new)
X = new_dum.values

size_of_sample = 10000
rows = np.random.choice(new.index.values, size=size_of_sample)
new_sample = new_dum.loc[rows]
X_sample = new_sample.values

model = IsolationForest()
model.fit(X_sample)
pred = model.predict(X_sample)
print pred[pred==-1].shape[0]*1./pred.shape[0]




pca = decomposition.PCA(n_components=10)
pca.fit(X_sample)
plt.plot(range(1,10+1),pca.explained_variance_ratio_)
plt.xlim(0,10+1)
plt.xticks(np.arange(1,10+1))
plt.ylabel('% of variance explained')
plt.show()

pca = decomposition.PCA(n_components=2)
pca.fit(X_sample) # better to use X, but it's too slow...
X_pca = pca.transform(X_sample)

num_clusters = np.arange(1,15+1)
scores = []
for num in num_clusters:
    print num
    #model = KModes(n_clusters=num,init='Huang',n_init=5,verbose=0,max_iter=500)
    #model = BayesianGaussianMixture(n_components=num, covariance_type='full')
    model = KMeans(n_clusters=num)
    model.fit(X_pca)
    #scores += [model.cost_]
    scores += [-model.score(X_pca)]
    
plt.plot(np.arange(1,15+1),scores)
plt.xlim(0,15+1)
plt.xticks(np.arange(1,15+1))
plt.xlabel('number of clusters')
plt.ylabel('loss')
plt.show()

colors = ['blue','red','green','darkorange','yellow','cyan','brown','purple','pink']

model = BayesianGaussianMixture(n_components=4, covariance_type='full')
model.fit(X_pca)
clf = model
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X_pca)
for i, color in enumerate(colors):
    if not np.any(Y_ == i):
        continue
    plt.scatter(X_pca[Y_ == i, 0], X_pca[Y_ == i, 1], .8, color=color)
plt.title('4 Clusters')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()

X_red = pca.transform(X)

probs = model.score_samples(X_red)
plt.plot(np.sort(probs))
plt.show()

print df['user'].loc[probs<-3].value_counts()

# <5% chance anomalies (user, # occurances)
# 264 occurances (using 2 clusters, get none using 4 clusters...)
# includes most users who logged in >=15 times, as recorded previously

anoms = df.loc[probs<-3]
#anoms['date'] = pd.to_datetime(anoms['date'])
print anoms['user'].value_counts()
anoms.to_csv('/Users/ryankingery/da-project-files/Ryan/suspicious_logons.csv')
    