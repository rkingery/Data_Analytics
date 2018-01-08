import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

#def clean(series):
#    out = []
#    series = series.dropna()
#    for value in series:
#        tmp_list = value.split(';')
#        for tmp in tmp_list:
#            out += [tmp]
#    return pd.Series(out)

#DATA_DIR = '/Users/ryankingery/da-project-files/Raw_Data/Second_Set/'

#df = pd.read_csv(DATA_DIR+'email_info.csv')
#
#new = df[['user','to','cc','bcc','from']]
#del(df)
#
#sent = new['from']
#received = pd.concat([clean(new['to']),clean(new['cc']),clean(new['bcc'])],axis=0)
#
#senders = sent.value_counts().index
#receivers = received.value_counts().index
#
#num_rows = senders.shape[0]
#num_cols = receivers.shape[0]
#adj_df = pd.DataFrame(np.zeros((num_rows,num_cols)),index=senders,columns=receivers)
#
#count = 1
#for sender in adj_df.index:
#    temp = new[new['from']==sender]
#    temp_rec = pd.concat([clean(temp['to']),clean(temp['cc']),clean(temp['bcc'])],axis=0)
#    for receiver in temp_rec:
#        adj_df[receiver][sender] += 1
#    print str(count)+' / '+str(num_rows)+' complete'
#    count += 1
#
#adj_df.to_csv('/Users/ryankingery/da-project-files/Ryan/network.csv')

# -------------------

#DATA_DIR = '/Users/ryankingery/da-project-files/Ryan/'
#df = pd.read_csv(DATA_DIR+'network.csv',index_col=0)
#num_rows = df.values.shape[0]
#num_cols = df.values.shape[1]
#
#plt.imshow(df.values[:100,:100],cmap='Greys',interpolation='nearest')
#plt.show()
#
#index = np.concatenate([df.index.values,df.columns.values],axis=0)
#index = pd.DataFrame(index).drop_duplicates().values.ravel()
#size = index.shape[0]
#adj_df = pd.DataFrame(np.zeros((size,size)),index=index,columns=index)
#
#count = 1
#for row in df.index.values:
#    for col in df.columns.values:
#        adj_df[col][row] = df[col][row]
#    if count % 1 == 0:
#        print count
#    count += 1
#
#adj_df.to_csv('/Users/ryankingery/da-project-files/Ryan/adj_matrix.csv')

# -------------------

DATA_DIR = '/Users/ryankingery/da-project-files/Ryan/'

adj_df = pd.read_csv(DATA_DIR+'adj_matrix.csv',index_col=0)
adj_matrix = (adj_df>0).values.astype(int)
fig = plt.imshow(adj_matrix[:2600,:])
plt.title('Adjacency Matrix')
plt.savefig(DATA_DIR+'adj_matrix.pdf',format='pdf')

index = adj_df.index.values
graph = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph())
#nx.draw_networkx(graph,node_size=1,width=0.1,with_labels=False)
#print nx.is_weakly_connected(graph)
#print nx.is_strongly_connected(graph)

pagerank = nx.pagerank(graph)