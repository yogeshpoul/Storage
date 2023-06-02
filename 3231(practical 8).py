#!/usr/bin/env python
# coding: utf-8

# In[1]:


#3272
#practical n0.8
import seaborn as sns


# In[2]:


df = sns.load_dataset('titanic')


# In[3]:


df


# In[4]:


df = df[['survived','class','sex','age','fare']]


# In[5]:


df


# In[6]:


sns.jointplot(x='age',y='fare',data=df)


# In[7]:


sns.jointplot(x='age',y='fare',data=df,hue='survived')


# In[8]:


sns.jointplot(x='age',y='fare',data=df,hue='class')


# In[9]:


sns.pairplot(df,hue='sex')


# In[10]:


sns.countplot(x=df['sex'])


# In[11]:


sns.countplot(x=df['class'])


# In[12]:


sns.barplot(x='sex',y='survived',data=df)


# In[13]:


sns.barplot(x='sex',y='survived',hue='class',data=df)


# In[15]:


sns.histplot(df['fare'])


# In[17]:


sns.kdeplot(df['fare'])


# In[ ]:




