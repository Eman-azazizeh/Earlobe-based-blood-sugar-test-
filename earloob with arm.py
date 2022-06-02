#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('earloobe_&_arm_data.csv')
data.head(10)a


# In[3]:


#convert the sex from categorical data to numerical data by using lambda 
data['Sex']= data['Sex'].apply(lambda x: 1 if x == 'male' else (0 if x =="female" else np.nan))
data.head(10)


# In[4]:


#split data
X=data.drop(['Arm'],axis=1).values

y=data['Arm'].values
print("X= \n",X)
print("_____________________________________________")
print("y=\n",y)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[6]:


data.describe()


# In[42]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2)


# In[43]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)


# In[44]:


y_pred=lr.predict(x_test)


# In[45]:


print(y_pred)


# In[46]:


print(y_test,'\n',y_pred)


# In[47]:


z=y_test-y_pred
print(z)


# In[48]:


from pandas.plotting import scatter_matrix
attributes = ["Arm",  "Earlob",
"Sex","Age"]
scatter_matrix(data[attributes], figsize=(12, 8))


# In[49]:


from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[50]:


from sklearn import metrics
#print result of mae
print("mae",metrics.mean_absolute_error(y_test,y_pred))
print("mse",metrics.mean_squared_error(y_test,y_pred))
print("rmse",(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))


# In[51]:


import seaborn as sns

sns.pairplot(data, x_vars=['Age','Sex','Earlob'], y_vars='Arm', size=7, aspect=0.7, kind='reg')


# In[ ]:





# In[ ]:




