# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:03:31 2015

@author: dataquanty
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


mat = pd.read_csv('bids.csv',sep=',')



mat['auction-device']=(mat['auction']+mat['device']).apply(lambda x: hash(x))
mat['auction-country']=(mat['auction']+mat['country']).apply(lambda x: hash(x))
mat['auction-ip']=(mat['auction']+mat['ip']).apply(lambda x: hash(x))
mat['auction-url']=(mat['auction']+mat['url']).apply(lambda x: hash(x))
mat['device-ip']=(mat['ip']+mat['device']).apply(lambda x: hash(x))
mat['device-country']=(mat['device']+mat['country']).apply(lambda x: hash(x))
mat['device-url']=(mat['device']+mat['url']).apply(lambda x: hash(x))
mat['device-merchandise']=(mat['device']+mat['merchandise']).apply(lambda x: hash(x))
mat['ip-url']=(mat['ip']+mat['url']).apply(lambda x: hash(x))
mat['ip-country']=(mat['ip']+mat['country']).apply(lambda x: hash(x))

# agg 1 : number of ... per bidder_id

def uniq(x):
    return np.size(np.unique(x))
    
def uniqRatio(x):
    return np.float(np.size(np.unique(x)))/np.size(x)


aggfunc = {'bid_id':uniq,
           'auction':uniq,
           'merchandise':uniq,
           'device':uniq,
           'country':uniq,
           'ip':uniq,
           'url':uniq,
           'auction-device':uniq,
           'auction-country':uniq,
           'auction-ip':uniq,
           'auction-url':uniq,
           'device-ip':uniq,
           'device-country':uniq,
           'device-url':uniq,
           'device-merchandise':uniq,
           'ip-url':uniq,
           'ip-country':uniq
           }

bidagg1 = mat.groupby('bidder_id').agg(aggfunc)
cols = bidagg1.columns.tolist()
fields = [x + 's' for x in cols]
bidagg1.columns = fields
bidagg1 = bidagg1.reset_index()

print bidagg1


## calc ratios
for f in fields:
    bidagg1[f+'_r']=bidagg1[f]/bidagg1['bid_ids']
    
bidagg1.drop('bid_ids_r',axis = 1,inplace=True)

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'pct_%s' % n
    return percentile_ 


def entropy(x):
    a = x.value_counts()/len(x)
    return -np.sum(a*np.log2(a))


def calcPop(field,mat):
    # agg 1 : popularity indices
    # 2 indices to make, 1 calc on all the data, another on the bidder_id agg
    pop = mat[field].value_counts()/len(mat)
    pop = pop.reset_index()
    pop.columns = [field,field + '_pop']
    mat = mat.merge(pop,left_on=field,right_on=field,how='left')
              
    aggfunc = {field + '_pop':[np.max,np.min,np.median,entropy],
               field:[entropy]
               }
    
    popagg = mat.groupby('bidder_id').agg(aggfunc)
    
    popagg.columns = ['_'.join(col).strip() for col in popagg.columns.values]
    popagg = popagg.reset_index()
    mat.drop(field + '_pop',inplace=True,axis=1)
    
    return popagg




fields = ['device','auction','merchandise','country','ip','url',
          'auction-device','auction-country','auction-ip','auction-url',
          'device-ip','device-country','device-url','device-merchandise',
          'ip-url','ip-country']

for f in fields:
    popagg = calcPop(f,mat)
    bidagg1 = bidagg1.merge(popagg,left_on='bidder_id',right_on='bidder_id',how='left')


#free memory
fields = ['auction-device','auction-country','auction-ip','auction-url',
          'device-ip','device-country','device-url','device-merchandise',
          'ip-url','ip-country']

mat.drop(fields,axis=1,inplace=True)



# Time 
# Time entropy to build ?
# Time agg function replace nan to improve

def countZeros(x):
    return np.sum(x==0)

def countZeros_r(x):
    return np.float(np.sum(x==0))/np.size(x)
    
    
def kurtosisAgg(x):
    return kurtosis(~np.isnan(x))

mat.sort(['bidder_id','time'],inplace=True)
mat['time_shift']=mat['time'].shift(-1)
mat['bidder_id_shift']=mat['bidder_id'].shift(-1)
mat['time_shift_c']=mat['time_shift'] * (mat['bidder_id']==mat['bidder_id_shift'])
mat['time_diff']=mat['time_shift_c']-mat['time']
mat['bidFolDay']=(mat['time_diff']>3*1e13)*1
mat['time_diff'] = mat['time_diff'].apply(lambda row : row if ((row >= 0) & (row<3.1e13)) else np.NaN)


aggfunc = {'time_diff':[np.max,np.min,np.median,countZeros_r,kurtosisAgg
#                        percentile(.25),percentile(.75),
                        ],
           'bidFolDay':[np.sum],
                        }
   
timeagg = mat.groupby('bidder_id').agg(aggfunc)
timeagg.columns = ['_'.join(col).strip() for col in timeagg.columns.values]
timeagg = timeagg.reset_index()

bidagg1 = bidagg1.merge(timeagg,left_on='bidder_id',right_on='bidder_id',how='left')

mat['time_diff_cl150'] = pd.cut(mat['time_diff'],150,labels=False)
mat['time_diff_cl20'] = pd.cut(mat['time_diff'],20,labels=False)
mat['time_diff_cl300'] = pd.cut(mat['time_diff'],300,labels=False)

aggfunc = {'time_diff_cl150':[entropy,uniqRatio],
           'time_diff_cl20':[entropy,uniqRatio],
           'time_diff_cl300':[entropy,uniqRatio],
            }

timeagg = mat.groupby('bidder_id').agg(aggfunc)
timeagg.columns = ['_'.join(col).strip() for col in timeagg.columns.values]
timeagg = timeagg.reset_index()
bidagg1 = bidagg1.merge(timeagg,left_on='bidder_id',right_on='bidder_id',how='left')


#Time 2
mat.sort(['bidder_id','auction','time'],inplace=True)
mat['time_shift']=mat['time'].shift(-1)
mat['bidder_id_shift']=mat['bidder_id'].shift(-1)
mat['auction_shift']=mat['auction'].shift(-1)
mat['time_shift_c']=mat['time_shift'] * ((mat['bidder_id']==mat['bidder_id_shift']) & (mat['auction']==mat['auction_shift']))
mat['time_diff_auct']=mat['time_shift_c']-mat['time']
#mat['bidFolDayAuct']=(mat['time_diff']>3*1e13)*1   # no one
mat['time_diff_auct'] = mat['time_diff_auct'].apply(lambda row : row if ((row >= 0) & (row<3.1e13)) else np.NaN)


aggfunc = {'time_diff_auct':[np.max,np.min,np.median,countZeros_r,kurtosisAgg
#                             percentile(.25),percentile(.75),
                             ]}
timeagg = mat.groupby('bidder_id').agg(aggfunc)
timeagg.columns = ['_'.join(col).strip() for col in timeagg.columns.values]
timeagg = timeagg.reset_index()
bidagg1 = bidagg1.merge(timeagg,left_on='bidder_id',right_on='bidder_id',how='left')


mat['time_diff_auct_cl50'] = pd.cut(mat['time_diff_auct'],50,labels=False)
mat['time_diff_auct_cl20'] = pd.cut(mat['time_diff_auct'],20,labels=False)
mat['time_diff_auct_cl100'] = pd.cut(mat['time_diff_auct'],100,labels=False)
mat['time_diff_auct_cl300'] = pd.cut(mat['time_diff_auct'],300,labels=False)
aggfunc = {'time_diff_auct_cl50':[entropy,uniqRatio],
           'time_diff_auct_cl20':[entropy,uniqRatio],
           'time_diff_auct_cl100':[entropy,uniqRatio],
           'time_diff_auct_cl300':[entropy,uniqRatio],
           }

timeagg = mat.groupby('bidder_id').agg(aggfunc)
timeagg.columns = ['_'.join(col).strip() for col in timeagg.columns.values]
timeagg = timeagg.reset_index()
bidagg1 = bidagg1.merge(timeagg,left_on='bidder_id',right_on='bidder_id',how='left')




#TopCat

def topCat(n):
    def topCat_(x):
        a = x.value_counts()
        a = a.index.tolist()[:n]
        a.sort()
        return str(a)
    topCat_.__name__ = 'topCat%s' % n
    return topCat_


aggfunc = {
#           'merchandise':[topCat(1),topCat(2),topCat(3)],
           'device':[topCat(1),topCat(2),topCat(3)],
           'country':[topCat(1),topCat(2),topCat(3)]}

   
topCategory = mat.groupby('bidder_id').agg(aggfunc)
topCategory.columns = ['_'.join(col).strip() for col in topCategory.columns.values]
topCategory = topCategory.reset_index()

bidagg1 = bidagg1.merge(topCategory,how='left',left_on='bidder_id',right_on='bidder_id')






def topCatAuctAgg(x):
    a = x.value_counts()
    a = a.index.tolist()[0]
    return a




#merchandise
aggfunc = {
           'merchandise':[topCatAuctAgg],
           'country':[topCatAuctAgg]}
topCatAuct = mat.groupby('auction').agg(aggfunc)
topCatAuct.columns = ['_'.join(col).strip() for col in topCatAuct.columns.values]
topCatAuct = topCatAuct.reset_index()

mat = mat.merge(topCatAuct,left_on='auction',right_on='auction',how='left')
mat['auctionCountry']= (mat['country_topCatAuctAgg']==mat['country'])*1
mat['auctionMerchandise']=(mat['merchandise_topCatAuctAgg']==mat['merchandise'])*1

aggfunc = {
           'auctionMerchandise':[np.mean,entropy],
           'auctionCountry':[np.mean,entropy]}

auctionAgg =  mat.groupby('bidder_id').agg(aggfunc)
auctionAgg.columns = ['_'.join(col).strip() for col in auctionAgg.columns.values]
auctionAgg = auctionAgg.reset_index()

bidagg1 = bidagg1.merge(auctionAgg,how='left',right_on='bidder_id',left_on='bidder_id')


#timeofday
mat['timeofday']=mat['time'].apply(lambda x : x%((6.4)*1e13))

aggfunc = {'timeofday':[np.max,np.min,np.median,
#                             percentile(.25),percentile(.75),
                             ]}
timeagg = mat.groupby('bidder_id').agg(aggfunc)
timeagg.columns = ['_'.join(col).strip() for col in timeagg.columns.values]
timeagg = timeagg.reset_index()
bidagg1 = bidagg1.merge(timeagg,left_on='bidder_id',right_on='bidder_id',how='left')

mat['timeofday_cl50'] = pd.cut(mat['timeofday'],50,labels=False)
mat['timeofday_cl20'] = pd.cut(mat['timeofday'],20,labels=False)
mat['timeofday_cl100'] = pd.cut(mat['timeofday'],100,labels=False)
mat['timeofday_cl300'] = pd.cut(mat['timeofday'],300,labels=False)
aggfunc = {'timeofday_cl50':[entropy,uniqRatio],
           'timeofday_cl20':[entropy,uniqRatio],
           'timeofday_cl100':[entropy,uniqRatio],
           'timeofday_cl300':[entropy,uniqRatio],
           }

timeagg = mat.groupby('bidder_id').agg(aggfunc)
timeagg.columns = ['_'.join(col).strip() for col in timeagg.columns.values]
timeagg = timeagg.reset_index()
bidagg1 = bidagg1.merge(timeagg,left_on='bidder_id',right_on='bidder_id',how='left')



#bidagg1 = pd.read_csv('bidagg.csv')

bidagg1.to_csv('bidagg.csv',index=False)



"""
#### Study
train = pd.read_csv('train.csv')
mat = mat.merge(train,left_on='bidder_id',right_on='bidder_id')
mat.sort(['bidder_id','time'],inplace=True)

mat['time_shift']=mat['time'].shift(-1)
mat['bidder_id_shift']=mat['bidder_id'].shift(-1)
mat['time_shift_c']=mat['time_shift'] * (mat['bidder_id']==mat['bidder_id_shift'])
mat['time_diff']=mat['time_shift_c']-mat['time']
mat['time_diff'] = mat['time_diff'].apply(lambda row : row if row >= 0 else np.NaN)

mat.drop(['time_shift','bidder_id_shift','time_shift_c'],inplace=True,axis=1)

mat[mat['outcome']==0].head(100000).to_csv('aaa.csv')
mat[mat['outcome']==1].head(100000).to_csv('bbb.csv')


###############################

mat[(mat['outcome']==0) & (mat['time_diff']!=0)]['time_diff'].plot(kind='hist',bins=100,logy=True)
mat[(mat['outcome']==1) & (mat['time_diff']!=0)]['time_diff'].plot(kind='hist',bins=100,logy=True)


mat = mat.reset_index()
mat = mat.set_index('outcome')
plt.figure()
mat[(mat['outcome']==0) & (mat['time_diff']!=0)]['time_diff'].hist(bins=100,alpha=0.5,log=True)
mat[(mat['outcome']==1) & (mat['time_diff']!=0)]['time_diff'].hist(bins=100,alpha=0.5,log=True)
plt.show()
mat[(mat['outcome']==0) & (mat['time_diff']!=0)][['outcome','time_diff']].hist(by='outcome',bins=100,alpha=0.5,log=True)

mat[(mat['outcome']==0) & (mat['time_diff']!=0)].groupby(['outcome','time_diff']).plot()

plt.figure()
df.plot(kind='hist',bins=100,logy=True,alpha=0.5,stacked=True)

mat['time'].hist(bins=1000,alpha=0.5,log=True)

def countsup(x):
    return np.sum(x>3*1e13)

aggfunc={'time_diff':countsup}

aggtime = mat.groupby('bidder_id').agg(aggfunc)


mat['timecat']=pd.cut(mat['time']/1e13,100)
mat.groupby('timecat')['bid_id'].count().to_csv('timecat.csv')
pd.DataFrame(np.unique(mat['timecat'])).to_csv('timecat.csv')


for i in range(10):
    mat['timeofday']=mat['time'].apply(lambda x : x%((6+i/10)*1e13))
    print 6+i/10
    print (mat['timeofday'].max()-mat['timeofday'].min())/1e13

mat['timeofday']=mat['time'].apply(lambda x : x%((6.4)*1e13))

"""
