#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np 
import pandas as pd 
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[51]:


df=pd.read_csv('parkinsons.data')
df.head()


# In[52]:


df.columns


# In[53]:


data=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,df.columns=='status'].values


# In[54]:


scaler=MinMaxScaler()
x=scaler.fit_transform(data)
y=labels


# In[55]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=150)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=100)


# In[47]:


rf.fit(xtrain,ytrain)


# In[48]:


predictions=rf.predict(xtest)


# In[49]:


from sklearn.metrics import accuracy_score
print(accuracy_score(predictions,ytest))


# In[57]:


from sklearn.model_selection import GridSearchCV
print(rf.get_params())


# In[71]:


n_estimators = [int(x) for x in np.linspace(start=200,stop=2000,num=5)]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
min_samples_split = [5, 10]
min_samples_leaf = [2, 4]
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# In[72]:


grid=GridSearchCV(RandomForestClassifier(),random_grid,refit=True,verbose=4,cv=3)


# In[73]:


grid.fit(xtrain,ytrain)


# In[74]:


grid.best_estimator_


# In[75]:


grid.best_params_


# In[79]:


best_model=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=85, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
best_model.fit(xtrain,ytrain)
b_predictions=best_model.predict(xtest)
print(accuracy_score(b_predictions,ytest))


# In[80]:


# best model has the same accuracy as the original model 
# more extensive parameters required in the search grid

import pickle 
pickle.dump(best_model,open('rf_model_v1','wb'))


# In[ ]:




