#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns


# In[2]:


df = sns.load_dataset('iris')


# In[3]:


df


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


sns.pairplot(df,hue='species')


# In[8]:


sns.pairplot(df)


# In[9]:


sns.pairplot(df,hue='species',diag_kind='hist')


# In[11]:


sns.histplot(df['sepal_length'],kde=True)


# In[12]:


sns.histplot(df['sepal_width'],kde=True)


# In[13]:


sns.kdeplot(df['sepal_width'])


# In[14]:


sns.histplot(df['petal_length'],kde=True)


# In[15]:


sns.kdeplot(df['petal_width'])


# In[17]:


sns.boxplot(x=df['sepal_length']);


# In[18]:


sns.boxplot(x=df['petal_length']);


# In[19]:


sns.boxplot(x=df['petal_width'])


# In[ ]:




