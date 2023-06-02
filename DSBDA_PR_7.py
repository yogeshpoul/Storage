#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install nltk -U')
get_ipython().system('pip install bs4 -U')


# In[6]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# In[7]:


import nltk


# In[8]:


para='Rajgad (literal meaning Ruling Fort) is a Hill region fort situated in the Pune district of Maharashtra, India. Formerly known as Murumbdev, the fort was the first capital of the Maratha Empire under the rule of Chhatrapati Shivaji for almost 26 years, after which the capital was moved to the Raigad Fort.[1] Treasures discovered from an adjacent fort called Torna were used to completely build and fortify the Rajgad Fort'


# In[9]:


print(para)


# In[10]:


para.split()


# In[11]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


# In[12]:


sent=sent_tokenize(para)


# In[13]:


sent[2]


# In[14]:


words=word_tokenize(para)


# In[15]:


words


# In[16]:


from nltk.corpus import stopwords


# In[17]:


swords=stopwords.words('english')


# In[18]:


swords


# In[19]:


x=[word for word in words if word not in swords]


# In[20]:


x


# In[23]:


x=[word for word in words if word.lower() not in swords]


# In[24]:


x


# In[25]:


from nltk.stem import PorterStemmer


# In[26]:


ps=PorterStemmer()


# In[27]:


ps.stem('working')


# In[28]:


y=[ps.stem(word) for word in x]


# In[29]:


y


# In[30]:


from nltk.stem import WordNetLemmatizer


# In[31]:


wnl=WordNetLemmatizer()


# In[32]:


wnl.lemmatize('workng', pos='v')


# In[33]:


nltk.download('omw-1.4')


# In[34]:


wnl.lemmatize('working', pos='v')


# In[35]:


print(ps.stem('went'))
print(wnl.lemmatize('went',pos='v'))


# In[36]:


z=[wnl.lemmatize(word,pos='v') for word in x]


# In[37]:


z


# In[38]:


import string


# In[39]:


string.punctuation


# In[40]:


t=[word for word in words if word not in string.punctuation]


# In[41]:


t


# In[42]:


from nltk import pos_tag


# In[43]:


pos_tag(t)


# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[46]:


tfidf= TfidfVectorizer()


# In[47]:


v=tfidf.fit_transform(t)


# In[48]:


v.shape


# In[49]:


import pandas as pd
pd.DataFrame(v)


# In[ ]:




