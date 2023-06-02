#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv('housing.csv')


# In[4]:


df


# In[5]:


df.head()


# In[8]:


x=df.drop('MEDV',axis=1)
y=df['MEDV']


# In[9]:


x.shape


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)


# In[12]:


x_train


# In[13]:


x_train.head()


# In[14]:


x_train.shape


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


regressor=LinearRegression()


# In[17]:


regressor.fit(x_train,y_train)


# In[18]:


regressor.coef_


# In[19]:


regressor.intercept_


# In[20]:


y_pred=regressor.predict(x_test)


# In[21]:


y_pred.shape


# In[22]:


result=pd.DataFrame({'Actual':y_test,'Producted':y_pred})


# In[23]:


result


# In[25]:


residual_error=abs(y_test-y_pred)


# In[28]:


residual_error


# In[29]:


residual_error;


# In[30]:


sum(residual_error)/len(residual_error)


# In[31]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred,y_test)


# In[32]:


regressor.score(x_test,y_test)


# In[33]:


from sklearn.metrics import r2_score 
r2_score(y_test,y_pred)


# In[35]:


new=[[0.7258,0,8.64,0,0.538,5.727,69.6,3.7965,4,307,22,391.95,11.28]]


# In[38]:


new

