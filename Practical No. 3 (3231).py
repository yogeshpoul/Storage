#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd


# In[25]:


df=pd.read_csv('student.csv')


# In[26]:


df


# In[27]:


df.mean()


# In[28]:


df.median()


# In[29]:


df.std()


# In[31]:


df.min()


# In[32]:


df.max()


# In[33]:


import numpy as np


# In[34]:


np.std(df['gpa'])


# In[35]:


gr1 = df.groupby('name')


# In[40]:


te = gr1.get_group('Jack')


# In[41]:


te.min()


# In[42]:


te.max()


# In[43]:


gr2 = df.groupby('id')


# In[44]:


gr2.groups


# In[49]:


tw = gr2.get_group(5225222)


# In[50]:


tw


# In[51]:


import seaborn as sns


# In[56]:


df=sns.load_dataset('Iris')


# In[53]:


df


# In[54]:


df=px.data.iris()


# In[55]:


df=pd.read_csv('Iris.csv')
df
df.describe()


# In[57]:


gr=df.groupby('species')


# In[59]:


se=gr.get_group('setosa')


# In[60]:


ve=gr.get_group('versicolor')


# In[61]:


vi=gr.get_group('virginica')


# In[62]:


se.shape


# In[63]:


ve.shape


# In[64]:


se.describe()


# In[65]:


ve.describe()


# In[66]:


vi.describe()


# In[ ]:




