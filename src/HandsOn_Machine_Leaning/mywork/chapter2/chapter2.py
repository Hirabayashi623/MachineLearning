
# coding: utf-8

# In[2]:


import os
import tarfile
from six.moves import urllib


# In[14]:


DOWNLAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL  = DOWNLAD_ROOT + 'datasets/housing/housing.tgz'


# ## ネットワーク上の圧縮ファイルをローカルに保存する

# In[15]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[16]:


fetch_housing_data()


# ## CSVファイルを読込む（配列として扱える）

# In[22]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


# ## データ構造を確認する

# In[23]:


housing = load_housing_data()
housing.head()


# ## データの情報の確認
# * info()メソッドを使うことで以下のことが確認できる
#     * 総行数
#     * 各属性のタイプ
#     * Nullではない数

# In[25]:


housing.info


# * 連想配列のように行を選択可能
# * value_counts関数によって個数の集計をとれる

# In[30]:


housing['ocean_proximity'].value_counts()


# In[33]:


import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
plt.show()


# # テストセットを作る
# * テストセットは全体の２０％ほどを確保する

# In[43]:


import numpy as np
def split_train_test(data, test_ratio=0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices  = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]


# In[44]:


train_set, test_set = split_train_test(housing)
print('trains:', len(train_set), ', tests: ', len(test_set))

