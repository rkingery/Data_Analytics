import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import decomposition
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.signal import medfilt, wiener
np.random.seed(42)

DATA_DIR = '/Users/ryankingery/da-project-files/Raw_Data/Second_Set/'

df = pd.read_csv(DATA_DIR+'device_info.csv')

new = df.drop(['id','date'],axis=1)#.reshape(-1,1)
new_dum = pd.get_dummies(new)
X = new_dum.values

size_of_sample = 50000
rows = np.random.choice(new.index.values, size=size_of_sample)
new_sample = new_dum.loc[rows]
X_sample = new_sample.values

model = IsolationForest()
model.fit(X_sample)
pred = model.predict(X)
print pred[pred==-1].shape[0]*1./pred.shape[0]


pca = decomposition.PCA(n_components=10)
pca.fit(new_sample.values)
X_sample = pca.transform(X_sample)
plt.plot(range(1,10+1),pca.explained_variance_ratio_)
plt.xlim(0,10+1)
plt.xticks(np.arange(1,10+1))
plt.ylabel('% of variance explained')
plt.show()

num_clusters = np.arange(1,15+1)
scores = []
for num in num_clusters:
    print num
    #model = KModes(n_clusters=num,init='Huang',n_init=5,verbose=0,max_iter=500)
    model = BayesianGaussianMixture(n_components=num, covariance_type='full')
    #model = KMeans(n_clusters=num)
    model.fit(X_sample)
    #scores += [model.cost_]
    scores += [-model.score(X_sample)]
    
plt.plot(np.arange(1,15+1),scores)
plt.xlim(0,15+1)
plt.xticks(np.arange(1,15+1))
plt.xlabel('number of clusters')
plt.ylabel('loss')
plt.show()

colors = ['blue','red','green','darkorange','yellow','cyan','brown','purple','pink']

model = BayesianGaussianMixture(n_components=4, covariance_type='full')
model.fit(X_sample)
clf = model
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X_sample)
for i, color in enumerate(colors):
    if not np.any(Y_ == i):
        continue
    plt.scatter(X_sample[Y_ == i, 0], X_sample[Y_ == i, 1], .8, color=color)
plt.title('4 Clusters')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()

X_red = pca.transform(X)

probs = model.score_samples(X_red)
plt.hist(probs)
plt.show()

anoms = df.loc[probs<-3]
#anoms['date'] = pd.to_datetime(anoms['date'])
print anoms['user'].value_counts()
anoms.to_csv('/Users/ryankingery/da-project-files/Ryan/suspicious_devices.csv')

# <5% chance anomalies (user, # occurances)
#DLM0051    7533
#BAL0044       4
#EIS0041       2
#GTD0219       2
#MOS0047       2
#BSS0369       2
#MPM0220       2
#AJF0370       2
#HSB0196    7595

#['DLM0051','BAL0044','EIS0041','GTD0219','MOS0047','BSS0369','MPM0220','AJF0370','HSB0196']

#files = pd.read_csv(DATA_DIR+'file_info.csv')
#files['date'] = pd.to_datetime(files['date'])

#users = list(anoms['user'].value_counts().index.values)
#users.remove('HSB0196')
#users += ['DLM0051']

#tmp = pd.DataFrame(columns=files.columns)
#for user in users:
#    dates = anoms[anoms['user']==user]['date'].dt.date.value_counts().index.values
#    for date in dates:
#        tmp = pd.concat([tmp,files[np.logical_and(files['user']==user,files['date'].dt.date==date)]])
#tmp.to_csv('/Users/ryankingery/da-project-files/Ryan/suspicious_devices.csv')
    