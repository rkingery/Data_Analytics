import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import decomposition
#from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from scipy.signal import medfilt, wiener
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import scale
np.random.seed(42)

DATA_DIR = '/Users/ryankingery/da-project-files/Raw_Data/Second_Set/'
MY_DIR = '/Users/ryankingery/da-project-files/Ryan/'

logon_df = pd.read_csv(DATA_DIR+'logon_info.csv')
logon_df.drop(['id','pc'],axis=1,inplace=True)
logon_df['date'] = pd.to_datetime(logon_df['date'],infer_datetime_format=True)
logon_df['date'] = logon_df['date'].dt.hour
#logon_df['date'] = logon_df['date'].apply(lambda x:x.split(' ')[0])
logon_df.sort_values(['user','date'],inplace=True)
#logon_df = logon_df.reindex(columns=['user','date','pc','activity'])
tmp_0 = logon_df.groupby(['user','date']).count().reset_index()
tmp_0 = tmp_0[['user','date','activity']]
tmp_0 = tmp_0.rename(columns={'activity':'logs'})
tmp_0.to_csv(MY_DIR+'log_counts.csv')
del(logon_df)
print 'logon_df done'

device_df = pd.read_csv(DATA_DIR+'device_info.csv')
device_df.drop(['id','pc'],axis=1,inplace=True)
device_df['date'] = pd.to_datetime(device_df['date'],infer_datetime_format=True)
device_df['date'] = device_df['date'].dt.hour
#device_df['date'] = device_df['date'].apply(lambda x:x.split(' ')[0])
device_df.sort_values(['user','date'],inplace=True)
tmp_1 = device_df.groupby(['user','date']).count().reset_index()
#tmp_1.drop('pc',axis=1,inplace=True)
tmp_1 = tmp_1.rename(columns={'activity':'devices'})
tmp_1['date'] = pd.to_datetime(tmp_1['date'])
#tmp_1['date'] = tmp_1['date'].dt.date
tmp_1.to_csv(MY_DIR+'device_counts.csv')
del(device_df)
print 'device_df done'

email_df = pd.read_csv(DATA_DIR+'email_info.csv')
email_df.drop(['pc','to','cc','bcc','content','from','id'],axis=1,inplace=True)
email_df['date'] = pd.to_datetime(email_df['date'],infer_datetime_format=True)
email_df['date'] = email_df['date'].dt.hour
#email_df['date'] = email_df['date'].apply(lambda x:x.split(' ')[0])
email_df.sort_values(['user'],inplace=True)
tmp_2 = email_df.groupby(['user','date']).count().reset_index()
tmp_2.drop('attachments',axis=1,inplace=True)
tmp_2 = tmp_2.rename(columns={'size':'emails_sent'})
tmp_2['size'] = email_df.groupby(['user','date']).sum().reset_index()['size']
tmp_2['attachments'] = email_df.groupby(['user','date']).sum().reset_index()['attachments']
#tmp_2['date'] = pd.to_datetime(tmp_2['date'])
tmp_2.to_csv(MY_DIR+'email_counts.csv')
del(email_df)
print 'email_df done'

file_df = pd.read_csv(DATA_DIR+'file_info.csv')
file_df.drop(['id','pc','content'],axis=1,inplace=True)
file_df['date'] = pd.to_datetime(file_df['date'],infer_datetime_format=True)
file_df['date'] = file_df['date'].dt.hour
#file_df['date'] = file_df['date'].apply(lambda x:x.split(' ')[0])
file_df.sort_values(['user'],inplace=True)
tmp_3 = file_df.groupby(['user','date']).count().reset_index()
tmp_3 = tmp_3.rename(columns={'filename':'device_files'})
#tmp_3['date'] = pd.to_datetime(tmp_3['date'])
tmp_3.to_csv(MY_DIR+'device_file_counts.csv')
del(file_df)
print 'file_df done'

#http_df = pd.read_csv('/Users/ryankingery/Desktop/http_stuff.csv',header=0,
#                      usecols=['date','user','url'],low_memory=True,
#                      parse_dates=['date'],infer_datetime_format=True)
http_df = pd.read_csv('/Users/ryankingery/Desktop/http_stuff_1.csv',low_memory=True,index_col=0)
http_df['date'] = pd.to_datetime(http_df['date'],infer_datetime_format=True)
http_df['date'] = http_df['date'].dt.hour
#http_df['date'] = http_df['date'].apply(lambda x:x.split(' ')[0])
http_df.sort_values(['user'],inplace=True)
tmp_4 = http_df.groupby(['user','date']).count().reset_index()
tmp_4 = tmp_4.rename(columns={'url':'http'})
tmp_4.to_csv(MY_DIR+'http_counts.csv')
del(http_df)
print 'http_df done'

users = tmp_0.user.unique()
hours = np.arange(0,24)
tmp1 = np.repeat(users,24).reshape((24000,1))
tmp2 = np.tile(np.arange(24),1000).reshape((24000,1))
data = np.concatenate([tmp1,tmp2],axis=1)
df = pd.DataFrame(data=data,columns=['user','date'])
df = df.merge(tmp_0,how='left',on=['user','date'])
df = df.merge(tmp_1,how='left',on=['user','date'])
df = df.merge(tmp_2,how='left',on=['user','date'])
df = df.merge(tmp_3,how='left',on=['user','date'])
df = df.merge(tmp_4,how='left',on=['user','date'])
#df = df.dropna(thresh=3)
df = df.fillna(value=0)
df['logs'] = df['logs'].astype(np.float64)
df = df.rename(columns={'date':'hours'})
df.to_csv(MY_DIR+'all_counts_by_hour.csv')

#-----------------------
df = pd.read_csv(MY_DIR+'all_counts_by_hour.csv')
df_list = [df[df.hours == i] for i in range(0,24)]

X = df.drop(['hours','user'],axis=1).values
X = scale(X)
X_list = [X[df.hours == i] for i in range(0,24)]

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

plt.scatter(X_pca[:,0],1./X_pca[:,1],s=0.1)
plt.show()

X_pca_list = [X_pca[df.hours == i] for i in range(0,24)]

for i in range(24):
    X_pca_i = X_pca_list[i]
    plt.scatter(X_pca_i[:,0],X_pca_i[:,1],s=0.8)
    plt.title('hour '+str(i))
    plt.show()
    
users = df.user.unique()

outliers_list = []
for i in range(24):
    X_pca_i = X_pca_list[i]
    #Xi = X[df.hours == i]
    #pca = decomposition.PCA(n_components=2)
    #pca.fit(Xi)
    #X_pca_i = pca.transform(Xi)
    model = IsolationForest(contamination=0.005)
    model.fit(X_pca_i)
    pred = model.predict(X_pca_i)
    outliers = X_pca_i[pred==-1,:]
    for outlier in outliers:
        outliers_list += [outlier]
    plt.scatter(X_pca_i[:,0],X_pca_i[:,1],s=.8,color='blue')
    plt.scatter(outliers[:,0],outliers[:,1],s=6.,color='red')
    plt.show()
outliers_list = np.array(outliers_list)#.reshape(len(outliers_list),7)

idx = []
for i,row in enumerate(X_pca[:]):
    for outlier in outliers_list[:]:
        if np.array_equal(row,outlier):
            idx += [i]
outlier_df = df.iloc[idx]



outliers_list = []    
for i in range(24):
    Xi = X[df.hours == i]
    model = IsolationForest(contamination=0.001)
    model.fit(Xi)
    pred = model.predict(Xi)
    outliers = Xi[pred==-1,:]
    for outlier in outliers:
        outliers_list += [outlier]
outliers_list = np.array(outliers_list).reshape(len(outliers_list),7)

idx = []
for i,row in enumerate(X[:]):
    for outlier in outliers_list[:]:
        if np.array_equal(row,outlier):
            idx += [i]
outlier_df = df.iloc[idx]

#X = df.drop(['hours','user','size','attachments'],axis=1).values
#count = 0
#outliers_list = []
#for user in users:
#    count += 1
#    print str(count)+': '+str(user)
#    for i in range(24):
#        X_user = X[np.logical_and(df.user.values==user,df.hours.values==i)]
#        #print X_user
#        if X_user.shape[0] != 1:
#            model = IsolationForest(contamination=0.01)
#            model.fit(X_user)
#            pred = model.predict(X_user)
#            outliers = X_user[pred==-1,:]
#            for outlier in outliers:
#                outliers_list += [outlier]
#            plt.scatter(X_user[:,0],X_user[:,1],s=.8,color='blue')
#            plt.scatter(outliers[:,0],outliers[:,1],s=2.,color='red')
#            plt.show()
#outliers_list = np.array(outliers_list)#.reshape(len(outliers_list),7)
#
#idx = []
#for i,row in enumerate(X[:]):
#    for outlier in outliers_list[:]:
#        if np.array_equal(row,outlier):
#            idx += [i]
#outlier_df = df.iloc[idx]