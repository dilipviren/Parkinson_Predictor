#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np 
import pandas as pd 
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[45]:


df=pd.read_csv('parkinsons.data')
df.head()


# In[46]:


df.columns


# In[47]:


labels=df.loc[:,df.columns=='status'].values
data=df.loc[:,df.columns!='status'].values[:,1:]


# In[48]:


scaler=MinMaxScaler()
x=scaler.fit_transform(data)
y=labels


# In[49]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=100)


# In[50]:


from sklearn.svm import SVC
svm=SVC()
svm.fit(xtrain,ytrain)


# In[51]:


predictions=svm.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,predictions))


# In[52]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[53]:


grid.fit(xtrain,ytrain)


# In[54]:


grid.best_params_


# In[55]:


grid.best_estimator_


# In[56]:


g_predictions=grid.predict(xtest)


# In[57]:


best_model=SVC(C=1000, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)


# In[58]:


best_model.fit(xtrain,ytrain)
b_predictions=best_model.predict(xtest)
print(accuracy_score(ytest,b_predictions))


# In[59]:


import pickle 
pickle.dump(best_model,open('svm_predictor','wb'))


# In[ ]:




