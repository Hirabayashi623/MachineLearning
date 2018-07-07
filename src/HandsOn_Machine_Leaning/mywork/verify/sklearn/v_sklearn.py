
# coding: utf-8

# In[7]:


import numpy as np


# # StratifiedShuffleSplit
# * 引数：
#     1. n_splits：分割数（分割をする回数）
#     2. test_size：データ全体に対するテストデータの割合
#     3. random_state：乱数の初期化のための値
# ## StratifiedShuffleSplit.split
# * 引数
#     1. 分割対象のデータ
#     2. データのグループ（分割時にグループの割合が保たれる）
# * 戻り値
#     * JavaでいうIterator的なオブジェクトが返却される
#     * 値の取得にはFor文を使う必要あり（それ以外の方法もある？）

# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[23]:


data = np.array(['A', 'B', 'a', 'b'])
group = np.array([0, 0, 1, 1])
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)


# In[24]:


for train_index, test_index in sss.split(data, group):
    print('%s %s %s %s' % (train_index, data[train_index], test_index, data[test_index]))

