import numpy as np 
import pandas as pd 
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df=pd.read_csv('parkinsons.data')
df.head()

labels=df.loc[:,df.columns=='status'].values
data=df.loc[:,df.columns!='status'].values[:,1:]

scaler=MinMaxScaler()
x=scaler.fit_transform(data)
y=labels


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=100)
print(xtrain.shape[0],xtest.shape[0])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

model=Sequential()


model.add(Dense(50,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=xtrain,y=ytrain,epochs=400,verbose=0)

# measuring model performance

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(xtest)
print(classification_report(ytest,predictions))
print(confusion_matrix(ytest,predictions))

# saving the model

from tensorflow.keras.models import load_model

model.save('sequential_parkinsons_predictor.h5')





