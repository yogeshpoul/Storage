#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


df


# In[4]:


x=df[['Age', 'EstimatedSalary']]

y=df['Purchased']


# In[5]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,random_state=0,test_size=0.33)


# In[10]:


x_train


# In[11]:


y_train


# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


import seaborn as sns
sns.countplot(x=y)


# In[14]:


y.value_counts()


# In[15]:


classifier = LogisticRegression()


# In[16]:


classifier.fit(x_train,y_train)


# In[17]:


y_pred = classifier.predict(x_test)


# In[18]:


y_train.shape


# In[19]:


x_train.shape


# In[20]:


y_pred


# In[21]:


y_test


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


plt.xlabel('Age')
plt.ylabel('Salary')
plt.scatter(x['Age'],x['EstimatedSalary'],c=y)


# In[24]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[25]:


pd.DataFrame(x_scaled).describe()


# In[26]:


plt.xlabel('Age')
plt.ylabel('Salary')
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=y)


# In[27]:


from sklearn.metrics import confusion_matrix


# In[28]:


confusion_matrix(y_test,y_pred)


# In[29]:


y_test.value_counts()


# In[30]:


from sklearn.metrics import plot_confusion_matrix


# In[31]:


plot_confusion_matrix(classifier,x_test,y_test)


# In[32]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[33]:


from sklearn.metrics import classification_report


# In[34]:


print(classification_report(y_test,y_pred))


# In[35]:


new1=[[26,34000]]
new2=[[57,138000]]


# In[36]:


classifier.predict(scaler.transform(new1))


# In[37]:


classifier.predict(scaler.transform(new2))


# In[ ]:




