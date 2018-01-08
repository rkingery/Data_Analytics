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
logon_df['date'] = pd.to_datetime(logon_df['date'])#,format='%m/%d/%Y',exact=False)
logon_df['date'] = logon_df['date'].dt.date
#logon_df['date'] = logon_df['date'].apply(lambda x:x.split(' ')[0])
logon_df.sort_values(['user','date'],inplace=True)
#logon_df = logon_df.reindex(columns=['user','date','pc','activity'])
tmp_0 = logon_df.groupby(['user','date']).count().reset_index()
tmp_0 = tmp_0[['user','date','activity']]
tmp_0 = tmp_0.rename(columns={'activity':'logs'})
tmp_0.to_csv(MY_DIR+'log_counts.csv')
del(logon_df)

device_df = pd.read_csv(DATA_DIR+'device_info.csv')
device_df.drop(['id','pc'],axis=1,inplace=True)
device_df['date'] = pd.to_datetime(device_df['date'],format='%m/%d/%Y',exact=False)
#device_df['date'] = device_df['date'].dt.date
#device_df['date'] = device_df['date'].apply(lambda x:x.split(' ')[0])
device_df.sort_values(['user','date'],inplace=True)
tmp_1 = device_df.groupby(['user']).count().reset_index()
#tmp_1.drop('pc',axis=1,inplace=True)
tmp_1 = tmp_1.rename(columns={'activity':'devices'})
tmp_1['date'] = pd.to_datetime(tmp_1['date'])
#tmp_1['date'] = tmp_1['date'].dt.date
tmp_1.to_csv(MY_DIR+'device_counts.csv')
del(device_df)

email_df = pd.read_csv(DATA_DIR+'email_info.csv')
email_df.drop(['pc','to','cc','bcc','content','from'],axis=1,inplace=True)
email_df['date'] = pd.to_datetime(email_df['date'],format='%m/%d/%Y',exact=False)
#email_df['date'] = email_df['date'].dt.date
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

file_df = pd.read_csv(DATA_DIR+'file_info.csv')
file_df.drop(['id','pc','content'],axis=1,inplace=True)
file_df['date'] = pd.to_datetime(file_df['date'],format='%m/%d/%Y',exact=False)
#file_df['date'] = file_df['date'].dt.date
#file_df['date'] = file_df['date'].apply(lambda x:x.split(' ')[0])
file_df.sort_values(['user'],inplace=True)
tmp_3 = file_df.groupby(['user','date']).count().reset_index()
tmp_3 = tmp_3.rename(columns={'filename':'device_files'})
#tmp_3['date'] = pd.to_datetime(tmp_3['date'])
tmp_3.to_csv(MY_DIR+'device_file_counts.csv')
del(file_df)

#http_df = pd.read_csv('/Users/ryankingery/Desktop/http_stuff.csv',header=0,
#                      usecols=['date','user','url'],low_memory=True,
#                      parse_dates=['date'],infer_datetime_format=True)
http_df = pd.read_csv('/Users/ryankingery/Desktop/http_stuff_1.csv',low_memory=True,index_col=0)
http_df['date'] = pd.to_datetime(http_df['date'],format='%m/%d/%Y',exact=False)
#http_df['date'] = http_df['date'].dt.date
#http_df['date'] = http_df['date'].apply(lambda x:x.split(' ')[0])
http_df.sort_values(['user'],inplace=True)
tmp_4 = http_df.groupby(['user','date']).count().reset_index()
tmp_4 = tmp_4.rename(columns={'url':'http'})
tmp_4.to_csv(MY_DIR+'http_counts.csv')
del(http_df)

df = pd.merge(tmp_0,tmp_1,how='left',on=['user','date'])
df = df.merge(tmp_2,how='left',on=['user','date'])
df = df.merge(tmp_3,how='left',on=['user','date'])
df = df.merge(tmp_4,how='left',on=['user','date'])
df = df.fillna(value=0)
df['logs'] = df['logs'].astype(np.float64)
df.to_csv(MY_DIR+'all_counts.csv')

# ----------------------
series=df[df['user']=='AAE0190'][['date','logs']]
#series=tmp_2.drop(['user','attachments','emails_sent'],1)
series['date'] = pd.to_datetime(series['date'])
series.set_index('date', inplace=True)
series = series.groupby(pd.TimeGrouper(level='date', freq='1D'))['logs'].agg('sum')   
series.dropna(inplace=True)
series = series.to_frame().reset_index()

series.plot()
plt.show()

decomp = seasonal_decompose(series['logs'].values, model='additive', freq=int(24*7))
decomp.plot()
plt.show()

tmp = pd.DataFrame(decomp.resid).dropna().values
resids = pd.DataFrame(decomp.resid).fillna(np.mean(tmp)).values
print resids.shape
print resids[np.abs(resids) > 3.*np.std(resids)].shape
idx = np.where(np.abs(resids) > 3.*np.std(resids))[0]
print series.iloc[idx,:]

users = tmp_0['user'].value_counts().index

#def STL(df,column,freq=24*7):
#    series = df.groupby(pd.TimeGrouper(level='date', freq='1D'))[column].agg('sum')
#    series.dropna(inplace=True)
#    series = series.to_frame().reset_index()
#    decomp = seasonal_decompose(series[column].values, model='additive', freq=freq)
#    return decomp
    
def find_outliers(decomp,series,column,num_sigmas=3.):
    tmp = pd.DataFrame(decomp.resid).dropna().values
    resids = pd.DataFrame(decomp.resid).fillna(np.mean(tmp)).values
    idx = np.where(np.abs(resids) > num_sigmas*np.std(resids))[0]
    return series.iloc[idx,:]

df = tmp_0.copy()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
column = 'logs'

for user in users[:]:
    user_df = df[df['user']==user]
    if user_df.shape[0] > 0:
        series = user_df.groupby(pd.TimeGrouper(level='date', freq='1D'))[column].agg('sum')
        series.dropna(inplace=True)
        series = series.to_frame().reset_index()
        try:
            decomp = seasonal_decompose(series[column].values, model='additive', freq=24*7)
        except:
            continue
        anoms = find_outliers(decomp,series,'logs',num_sigmas=10.)
    if anoms.shape[0] > 0:
        #print anoms.shape[0]
        if anoms.shape[0] < 100:
            print anoms
        #print str(user)+': '+str(anoms.shape[0])


# use general trend for each user?


#dates = logon_df['date'].dt.date.value_counts().index

#----------------------
df = pd.read_csv(MY_DIR+'all_counts.csv')

X = df.drop(['date','user'],axis=1).values
X = scale(X)

pca = decomposition.PCA(n_components=2,whiten=True)
pca.fit(X)
X_pca = pca.transform(X)

plt.scatter(X_pca[:,0],X_pca[:,1],s=0.1)
plt.show()

#model = LocalOutlierFactor(contamination=0.0001)
model = IsolationForest(contamination=0.001)
model.fit(X_pca)
pred = model.predict(X_pca)
print pred[pred==-1].shape[0]*1./pred.shape[0]

outliers = X_pca[pred==-1,:]

plt.scatter(X_pca[:,0],X_pca[:,1],s=.1,color='blue')
plt.scatter(outliers[:,0],outliers[:,1],s=.1,color='red')
plt.show()


