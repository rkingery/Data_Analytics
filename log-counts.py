# This script uses the logon_info.csv file to count the number of login attempts per
# user and prints login attempts per day exceeding some specified cutoff (in my case, 15)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = '/Users/ryankingery/da-project-files/Raw_Data/Second_Set/'


df = pd.read_csv(DATA_DIR+'logon_info.csv')
df['activity'] = df['activity'].replace({'Logon':1,'Logoff':0})
df['date'] = pd.to_datetime(df['date'])

pd.crosstab(index = df['activity'],columns='count').plot.bar()

# look for users who login a lot in a single day
user_counts = df['user'].value_counts()
cutoff = 18
tmp = pd.DataFrame([])
for user in user_counts.index:
    tmp = df[df['user']==user]
    date_counts = tmp['date'].dt.date.value_counts()
    for date in date_counts.index:
        if date_counts[date] >= cutoff:
            print 'User '+str(user)+' has '+str(date_counts[date])+\
            ' log attempts on '+str(date)

# look at logon/logoff ratio by date
plt.plot(df['activity'].groupby(by=df['date'].dt.date).apply(np.mean).values)
plt.title('Percent Logons by Date')
plt.xlabel('Date')
plt.ylabel('Percent Logons')
plt.show()

# look at total logs by from most users to least
plt.plot(df['user'].value_counts()[df['user'].value_counts()>1000].values)
plt.title('Total logins by ranked users')
plt.xlabel('ranked users')
plt.ylabel('total logins')
plt.show()

# look at total logs over time, segmented by 50 day periods (for easy viewing)
num_dates = len(df['date'].dt.date.value_counts())
for i in range(num_dates//50):
    plt.plot(df['activity'].groupby(by=df['date'].dt.date)\
             .apply(np.sum).values[50*i:50*i+50])
    plt.title('Logs for days '+str(50*i+1)+' through '+str(50*i+50))
    plt.show()
    
# looking at number of times PCs are used, most to least
plt.plot(df['pc'].value_counts().values)

# notes: 
# logs collected 1/2/2010 to 3/17/2011 (1 year and 2.5 months)
# of 1000 users, most have logged in >= 100 times
# all 1000 ids are unique (on purpose?)
# about 55% of total log activities are logons
# every day there are more login attempts than logoffs (52%-62% logons)
# users with >= 15 logs in a single day (# times): 
#   CBB0365 (10), BAL0044 (2), IBB0359 (6), EIS0041 (3), WDD0366 (2), MPM0220 (5),
#   MOS0047 (10), CCA0046 (1), JTM0223 (6), CSC0217 (1)
# no discernable anomalies in total logs per day over time except an expected
#   weekend/holiday dip trend and a slow decrease in max logs over time (1400->1200)
# You can see logs do dip over weekends and holiday periods, but never to zero
# there are clearly a few users who log a lot (over 3000 times in a year), 
#   while most don't really log over 1000 times in a year