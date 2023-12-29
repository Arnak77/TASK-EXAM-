#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")


# In[15]:


satrt=pd.read_csv(r"D:\NIT\DATASCIENCE\ARNAK TASK\exam 1\movies.csv")


# In[16]:


satrt


# In[17]:


satrt = satrt.dropna(how='all')


# In[18]:


satrt


# In[19]:


satrt.isnull().sum()


# In[20]:


satrt["RATING"] 


# In[21]:


satrt["RATING"].nunique()


# In[22]:


satrt["RATING"].unique()


# In[24]:


satrt["RATING"] .isnull().sum()


# In[27]:


satrt['RATING']=satrt['RATING'].fillna(np.mean(pd.to_numeric(satrt['RATING'])))


# In[28]:


satrt["RATING"] .isnull().sum()


# In[33]:


satrt["RATING"].unique()


# In[34]:


satrt['VOTES']=satrt['VOTES'].str.replace(r'\W','',regex=True)


# In[35]:


satrt['VOTES']=satrt['VOTES'].fillna(np.mean(pd.to_numeric(satrt['VOTES'])))


# In[ ]:





# In[36]:


satrt['RunTime']=satrt['RunTime'].fillna(np.mean(pd.to_numeric(satrt['RunTime'])))


# In[44]:


satrt.isnull().sum()


# In[46]:


satrt.drop(columns='Gross', inplace=True)


# In[48]:


satrt.columns


# In[49]:


satrt.isnull().sum()


# In[52]:


satrt.describe()


# In[53]:


satrt.info()


# In[54]:


satrt.dtypes


# In[55]:


satrt.head(2)


# In[64]:


satrt['MOVIES']=satrt['MOVIES'].astype('category')


# In[65]:


satrt['GENRE']=satrt['GENRE'].astype('category')


# In[66]:


satrt['ONE-LINE']=satrt['ONE-LINE'].astype('category')


# In[67]:


satrt['STARS']=satrt['STARS'].astype('category')


# In[68]:


satrt['RATING']=satrt['RATING'].astype('float')


# In[69]:


satrt['VOTES']=satrt['VOTES'].astype('int')


# In[70]:


satrt['RunTime']=satrt['RunTime'].astype('float')


# In[74]:


satrt.info()


# In[75]:


satrt.describe()


# In[76]:


Clean_Starts1=satrt.copy()


# In[77]:


Clean_Starts1.to_csv('Clean_Starts1.csv')


# In[78]:


import os
os.getcwd()


# In[ ]:





# In[ ]:




