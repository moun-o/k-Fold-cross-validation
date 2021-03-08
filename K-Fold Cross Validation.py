#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.options.mode.chained_assignment=None


# In[2]:


drop_cols=['room_type','city','zipcode','state','longitude','latitude','host_response_rate','host_acceptance_rate','host_response_rate','host_listings_count']

#read data
airbnb_list=pd.read_csv('paris_airbnb.csv')

#clean the column price, remove , and $
airbnb_list['price']=airbnb_list['price'].astype(str).str.replace('$','')
airbnb_list['price']=airbnb_list['price'].astype(str).str.replace(',','')

#convert price column to float
airbnb_list['price']=airbnb_list['price'].astype('float')

#delete the non float and non interesting colomuns
airbnb_list=airbnb_list.drop(drop_cols,axis=1)

#drop null rows
airbnb_list=airbnb_list.dropna(axis=0)


# In[3]:


#import cross validation tools
from sklearn.model_selection import KFold, cross_val_score

#import the KNN estimator
from sklearn.neighbors import KNeighborsRegressor


# In[4]:


#init features for training
features=airbnb_list[['accommodates','bedrooms','beds']]

#init the target feature
target=airbnb_list[['price']]


# In[5]:


k_max=30
for k in range(3,k_max,2):
    #init the kfold , 5 is the number of batchs, shuffle = true if you want to shuffle the data before split
    kf=KFold(k,shuffle=True, random_state=1)

    #init the KNN
    knn=KNeighborsRegressor()

    #Evaluate tge MSE on cross validation train, 
    errorMSE=cross_val_score(knn,features,target,scoring='neg_mean_squared_error',cv=kf)

    #get the RMSE
    errorRMSE=np.sqrt(np.absolute(errorMSE))

    #print the cross validation results
    print("k= ",k, "error is :" ,np.mean(errorRMSE))

