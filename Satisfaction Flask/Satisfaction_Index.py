#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
from time import strftime
import tensorflow as tf
import keras
import requests
import json
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.svm import SVC
#from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten, Conv2D, MaxPooling2D
from keras.layers import *
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
get_ipython().run_line_magic('reload_ext', 'tensorboard')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('employees (1) (1) (1) (2).csv')
data


# In[3]:


data.isnull().sum()


# In[4]:


data['last_evaluation'].fillna('0',inplace=True)


# In[5]:


data.isnull().sum()


# In[6]:


data['department'].fillna('support',inplace=True)


# In[7]:


data['satisfaction'].fillna('0.0',inplace=True)


# In[8]:


data['tenure'].fillna('0',inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


le = LabelEncoder()


# In[11]:


label = le.fit_transform(data['EmployeeName'])
label


# In[12]:


label = le.fit_transform(data['Agency'])
label


# In[13]:


label = le.fit_transform(data['fname'])
label


# In[14]:


label = le.fit_transform(data['lname'])
label


# In[15]:


data.drop("EmployeeName", axis=1, inplace=True)


# In[16]:


data.drop("Agency", axis=1, inplace=True)


# In[17]:


data.drop("fname", axis=1, inplace=True)


# In[18]:


data.drop("lname", axis=1, inplace=True)


# In[19]:


data["EmployeeName"] = label


# In[20]:


data["Agency"] = label


# In[21]:


data["fname"] = label


# In[22]:


data["lname"] = label


# In[23]:


data.head()


# In[24]:


data['status']=data['status'].map({'Employed':1,'Left':0})
data.head()


# In[25]:


data['salary']=data['salary'].map({'low':0,'medium':1,'high':2})
data.head()


# In[26]:


data['department']=data['department'].map({'product':0,'sales':1,'support':2,'temp':3,'IT':4,'admin':5,'engineering':6,'finance':7,'information_technology':8,'management':9,'marketing':10,'procurement':11})


# In[27]:


data.groupby('department')['satisfaction'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[28]:


data.groupby('n_projects')['satisfaction'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[29]:


data.groupby('salary')['satisfaction'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[30]:


data.groupby('tenure')['satisfaction'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[31]:


data.groupby('status')['satisfaction'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[ ]:





# In[ ]:





# In[32]:


data.corr().head()


# In[33]:


plt.figure(figsize=(16,10))
sns.heatmap(data.corr(), annot=True, annot_kws={"size": 14})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[34]:


plt.figure(figsize=(7,7))
sns.kdeplot(data['salary'],shade=True)
plt.show()


# In[35]:


plt.figure(figsize=(7,7))
sns.kdeplot(data['avg_monthly_hrs'],shade=True)
plt.show()


# In[36]:


sns.pairplot(data)


# In[37]:


data = data.drop(['Agency'],axis=1)


# In[38]:


data = data.drop(['EmployeeName'],axis=1)


# In[39]:


data = data.drop(['fname'],axis=1)


# In[40]:


data=data.drop(['lname'],axis=1)


# In[41]:


data = data.drop(['avg_monthly_hrs'],axis=1)


# In[42]:


data = data.drop(['last_evaluation'],axis=1)


# In[43]:


data.head()


# In[44]:


target = np.array(data.drop(['satisfaction'],1))
features = np.array(data['satisfaction'])


# In[45]:


target[0]


# In[46]:


x_train , x_test , y_train , y_test = train_test_split(target,features,test_size=0.25,random_state=42)

len(x_train)/len(features)


# In[47]:


print(f'Shape of x_train is {x_train.shape}')
print(f'Shape of x_test is {x_test.shape}')
print(f'Shape of y_train is {y_train.shape}')
print(f'Shape of y_test is {y_test.shape}')


# In[48]:


rf_model = RandomForestRegressor(random_state = 42).fit(x_train, y_train)


# In[49]:


y_pred = rf_model.predict(x_test)
rf_base = np.sqrt(mean_squared_error(y_test, y_pred))
rf_base


# In[50]:


knn_model = KNeighborsRegressor().fit(x_train, y_train)


# In[51]:


cart_model = DecisionTreeRegressor()
cart_model.fit(x_train, y_train)


# In[52]:


y_pred = cart_model.predict(x_test)
cart_base = np.sqrt(mean_squared_error(y_test, y_pred))
cart_base


# In[53]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])


# In[54]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(x_train)
lr_poly_regressor = LinearRegression()
lr_poly_regressor.fit(X_poly, y_train)

# Predicting Test Set
y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(x_test))


# In[55]:


from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

model_results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)


# In[56]:


from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_train, y_train)

# Predicting Test Set
y_pred = dt_regressor.predict(x_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

model_results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)


# In[57]:


## Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(x_train,y_train)

# Predicting Test Set
y_pred = rf_regressor.predict(x_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)


# In[58]:


from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()
xgb_regressor.fit(x_train, y_train)

# Predicting Test Set
y_pred = xgb_regressor.predict(x_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)


# In[59]:


##Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(x_train, y_train)

# Predicting Test Set
y_pred = gb_regressor.predict(x_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

model_results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)


# In[60]:


## Ada Boosting
from sklearn.ensemble import AdaBoostRegressor
ad_regressor = AdaBoostRegressor()
ad_regressor.fit(x_train, y_train)

# Predicting Test Set
y_pred = ad_regressor.predict(x_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

model_results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)


# In[61]:


#The Best Classifier
print('The best regressor is:')
print('{}'.format(results.sort_values(by='R2 Score',ascending=False).head(5)))


# In[62]:


import pickle
pickle.dump(gb_regressor,open('employee_satisfaction_1.pkl','wb'))


# In[63]:


model = pickle.load(open('employee_satisfaction_1.pkl','rb'))


# In[64]:


print(gb_regressor.predict([[6,4,0,1,5,43,1.4,567695]]))


# In[65]:


print(gb_regressor.predict([[5,2,0,1,2,23,3.4,367695]]))


# In[70]:


print(gb_regressor.predict([[1,1,0,0,2,33,4.4,347695]]))


# In[73]:


print('Satisfaction index of the employee is:',gb_regressor.predict([[4,4,0,2,2,23,4.4,747695]]))


# In[79]:


print('Satisfaction index of the employee is:',gb_regressor.predict([[4,8,0,0,1,23,1.4,47695]]))


# In[ ]:




