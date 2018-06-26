
# coding: utf-8

# In[3]:


import os
import pandas as pd


# CSVファイルのパスを定義

# In[5]:


csv_path = os.path.join('data', 'housing.csv')


# pandasを使ってデータロード

# In[9]:


housing = pd.read_csv(csv_path)


# In[11]:


housing[:5]


# In[13]:


housing.head()


# ## pandas.DataFrame.hist
# ### parameter
# * bin：柱の数を指定
# * figsize：図の大きさを指定

# In[22]:


import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
plt.show()

