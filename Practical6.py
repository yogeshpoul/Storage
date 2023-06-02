#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


df.shape


# In[6]:


df


# In[7]:


x=df.drop('Species',axis=1)


# In[8]:


y=df['Species']


# In[9]:


y.value_counts()


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train ,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)


# In[12]:


x_train.shape


# In[13]:


x_test.shape


# In[14]:


from sklearn.naive_bayes import GaussianNB


# In[15]:


clf= GaussianNB()


# In[16]:


clf.fit(x_train,y_train)


# In[17]:


y_pred=clf.predict(x_test)


# In[18]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# In[19]:


confusion_matrix(y_test,y_pred)


# In[25]:


plot_confusion_matrix(clf,x_test,y_test)


# In[21]:


accuracy_score(y_test,y_pred)


# In[22]:


clf.predict_proba(x_test)


# In[31]:


newl=[[4.5,2.9,3.1,0.4,0.8]]
clf.predict(newl)[0]


# In[33]:


newl=[[5.5,3.1,1.0,0.8,1.0]]
clf.predict(newl)[0]


# In[34]:


newl=[[6.5,3.3,4.9,1.8,1.7]]
clf.predict(newl)[0]


# In[27]:


print(classification_report(y_test,y_pred))


# In[ ]:




