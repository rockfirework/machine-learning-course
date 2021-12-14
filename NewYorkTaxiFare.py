#!/usr/bin/env python
# coding: utf-8

# 1、导入数据

# In[1]:


# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

print(os.listdir('../input'))


# In[2]:


train =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 10000)

test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


# 2、查看数据基本情况

# In[3]:


train.shape


# In[4]:


test.shape


# 训练集有100万行数据，8个字段。预测集有9914行数据，7个字段。

# In[5]:


train.head()


# 训练集train的描述性分析

# In[6]:


train.describe()


# In[7]:


train.isnull().sum().sort_values(ascending=False)


# In[8]:


test.isnull().sum().sort_values(ascending=False)


# In[9]:


train = train.drop(train[train['dropoff_latitude'].isnull()].index,axis=0) #删除10条含缺失值的数据


# In[10]:


train = train.drop(train[train['fare_amount']<0].index,axis=0)
train['fare_amount'].describe()


# In[11]:


train[train['passenger_count']>6]


# In[12]:


train = train.drop(train[train['passenger_count']>6].index,axis=0)
train['passenger_count'].describe()


# In[13]:


train['pickup_latitude'].describe()


# In[14]:


train['pickup_longitude'].describe()


# In[15]:


train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90]))
         .index,axis=0)
train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180]))
         .index,axis=0)
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90]))
         .index,axis=0)
train = train.drop(((train[train['dropoff_longitude']<-180])|(train[train['dropoff_longitude']>180])
         ).index,axis=0)
train.shape


# In[16]:


train.dtypes


# In[17]:


train['key']=pd.to_datetime(train['key'])
train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
test['key']=pd.to_datetime(test['key'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])


# In[18]:


train.dtypes


# In[19]:


test.dtypes


# In[20]:


def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  #radius of earth in kilometers
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['H_Distance'] = d
    return d


# In[21]:


haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


# In[22]:


train.head(10)


# In[23]:


test.head(10)


# In[24]:


train['H_Distance'].describe()


# In[25]:


train.loc[((train['pickup_latitude']==0)&(train['pickup_longitude']==0))&((train['dropoff_latitude']
          !=0) & (train['dropoff_longitude']!=0))& (train['fare_amount']==0)]


# In[26]:


train = train.drop(train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))
&((train['dropoff_latitude']!=0)& (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)].index, axis=0)


# In[27]:


train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))
&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0))&(train['fare_amount']==0)]


# In[28]:


train = train.drop(train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))
&((train['dropoff_latitude']==0)&(train['dropoff_longitude']==0)) & (train['fare_amount']==0)].index, axis=0)


# In[29]:


train = train.drop(train.loc[(train['H_Distance']>200)&(train['fare_amount']!=0)].index,axis=0)


# In[30]:


train.loc[(train['H_Distance']==0) & (train['fare_amount'] < 2.5)]


# In[31]:


train = train.drop(train.loc[(train['H_Distance']==0) & (train['fare_amount'] < 2.5)].index,axis=0)


# In[32]:


F3h0 = train.loc[(train['fare_amount']>3)&(train['H_Distance']==0)]


# In[33]:


F3h0['H_Distance'] = F3h0.apply(lambda row: ((row['fare_amount']-2.50)/1.56), axis=1)
train.update(F3h0)


# In[34]:


#用公式计算车费
#H1f0 = train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]
#H1f0['fare_amount'] = H1f0.apply(lambda row: (row['H_Distance']*1.56+2.5),axis=1)
#train.update(H1f0)


# In[35]:


data = [train,test]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour


# In[36]:


train.head()


# In[37]:


train = train.drop(['key','pickup_datetime'], axis = 1)
test = test.drop(['key','pickup_datetime'], axis = 1)


# In[38]:


#导入包
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'],y=train['fare_amount'],s=1.5)
plt.title('Relationship between number of passengers and price')
plt.xlabel('Number of passengers')
plt.ylabel('Price')
plt.show()


# In[39]:


plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'],y=train['fare_amount'],s=1.5)
plt.title('Relationship between time and price')
plt.xlabel('time')
plt.ylabel('Price')
plt.show()


# In[40]:


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'],y=train['fare_amount'],s=1.5)
plt.title('Relationship between date and price')
plt.xlabel('date')
plt.ylabel('Price')
plt.show()


# In[41]:


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day of week'],y=train['fare_amount'],s=1.5)
plt.title('Relationship between week and price')
plt.xlabel('week')
plt.ylabel('Price')
plt.show()


# In[42]:


x_train = train.iloc[:,train.columns!='fare_amount'] #测试集 x
y_train = train['fare_amount'].values #测试集 y
x_test = test #预测数据


# In[43]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[44]:


rf.fit(x_train, y_train)


# In[45]:


rf_predict = rf.predict(x_test)
rf_predict


# In[46]:


submission = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
submission['fare_amount'] = rf_predict
submission.to_csv('./submission_1.csv', index=False)

