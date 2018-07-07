
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# CSVファイルのパスを定義

# In[2]:


csv_path = os.path.join('data', 'housing.csv')


# pandasを使ってデータロード

# In[3]:


housing = pd.read_csv(csv_path)


# In[4]:


housing[:5]


# In[5]:


housing.head()


# ## pandas.DataFrame.hist
# ### parameter
# * bin：柱の数を指定
# * figsize：図の大きさを指定

# In[6]:


import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
plt.show()


# # 特定のグラフを描画
# 列名をしていする（pandas.DataFrame[列名]）

# In[7]:


housing['population'].hist(bins=50)
plt.show()


# ### padas.DataFrame.where

# In[8]:


data = pd.DataFrame({'A':[1,2,3,4,5]})
print(data)


# In[9]:


data['B'] = data['A'].where(data['A']>3)
print(data)


# * 第二引数を指定するとNaNのところに指定した値が埋め込まれる

# In[10]:


data['C'] = data['A'].where(data['A']>3, 10)
print(data)


# * 「inplace=True」とすると元のオブジェクトを上書きする

# In[11]:


data['D'] = data['A'] # 列Aをコピー
data['D'].where(data['D']>3, 10, inplace=True)
print(data)


# # 相関を求める

# In[12]:


data1 = np.random.rand(100)
data2 = np.random.rand(100)
data3 = np.random.rand(100)
data = pd.DataFrame({'data1':data1, 'data2': data2, 'data3': data3})
print(data.corr())


# In[13]:


from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()


# ## データを自分で作成する

# In[16]:


data_sample = pd.DataFrame({'sample': [1,2,3]})
print(data_sample)


# ### 任意の位置の値を取得・変更する

# In[24]:


test = housing.copy()
test['ocean_proximity'].value_counts()


# In[28]:


test['ocean_proximity'].loc[:, '<1H OCEAN'] = 0

