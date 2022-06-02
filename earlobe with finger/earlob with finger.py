#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


data=pd.read_csv('finger_&_earlob_data.csv')
data.head(10)


# In[4]:


#convert the sex from categorical data to numerical data by using lambda 
data['Sex']= data['Sex'].apply(lambda x: 1 if x == 'male' else (0 if x =="female" else np.nan))
data.head(10)


# In[6]:


#split data
X=data.drop(['Finger'],axis=1).values

y=data['Finger'].values
print("X= \n",X)
print("_____________________________________________")
print("y=\n",y)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[8]:


data.describe()


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[10]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)


# In[11]:


y_pred=lr.predict(x_test)


# In[12]:


print(y_pred)


# In[13]:


print(y_test,'\n',y_pred)


# In[14]:


z=y_test-y_pred
print(z)


# In[16]:


from pandas.plotting import scatter_matrix
attributes = ["Finger",  "Earlob",
"Sex","Age"]
scatter_matrix(data[attributes], figsize=(12, 8))


# In[17]:


from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[18]:


from sklearn import metrics
#print result of mae
print("mae",metrics.mean_absolute_error(y_test,y_pred))
print("mse",metrics.mean_squared_error(y_test,y_pred))
print("rmse",(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))


# In[19]:


import seaborn as sns

sns.pairplot(data, x_vars=['Age','Sex','Earlob'], y_vars='Finger', size=7, aspect=0.7, kind='reg')


# In[ ]:




