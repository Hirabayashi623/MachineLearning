
# coding: utf-8

# # データ準備部分の実装

# In[1]:


import numpy as np
import myutil
import os
import tarfile
from six.moves import urllib
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


# 定数の定義
DOWNLAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_NAME = 'housing.tgz'
HOUSING_URL  = DOWNLAD_ROOT + 'datasets/housing/housing.tgz'


# In[3]:


# データのダウンロード
myutil.downloadAfterCheck()


# In[4]:


# データの読込
housing = myutil.load_housing_data('datasets/housing/housing.csv')
housing.head(5)
housing.hist(bins=50, figsize=(20,15))
plt.plot()


# In[5]:


df = myutil.DataFactory()
# 訓練データとテストデータの分割
data_train, data_test = df.split_dataset(housing)
# ラベルとデータの分割
data_label, data_train = df.split_label(data_train, 'median_house_value')


# In[6]:


list(data_train)


# In[7]:


ls = list(data_train)
ls.remove('ocean_proximity')
ls


# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# In[9]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_columns):
        self.num_columns = num_columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.num_columns].values


# In[10]:


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # 転置行列にする
        T = X.T
        # 返却用の配列を定義
        X_trs = []
        # 文字列属性分、ワンホットエンコードを実施
        for col in np.arange(T.shape[0]):
            # 属性に含まれる値の分、列を追加＆バイナリ化
            for val in set(T[col]):
                tmp = T[col].copy()
                tmp[tmp==val] = 1
                tmp[tmp!=1] = 0
                X_trs.append(tmp)
        # 転置し直す
        X_trs = np.array(X_trs)
        return X_trs.T


# In[11]:


# 数値属性について実施するパイプラインの定義
str_columns = ['ocean_proximity']
num_columns = list(data_train)
for str_column in str_columns:
    num_columns.remove(str_column)
num_columns


# In[12]:


pipeline_num = Pipeline([
    ('selector'  , DataFrameSelector(num_columns)),  # 数値属性だけ取り出す
    ('imputer'   , Imputer(strategy='median')),      # 欠損値を中央値で補う
    ('std_scalar', StandardScaler()),                # 標準化する
])


# In[13]:


pipeline_str = Pipeline([
    ('selector'   , DataFrameSelector(str_columns)), # 文字列属性だけ取り出す
    ('str_encoder', OneHotEncoder()),                # ワンホットエンコード
])


# In[14]:


from sklearn.pipeline import FeatureUnion
# ２つのパイプラインで変換したデータを結合させるパイプライン
pipeline_main = FeatureUnion(transformer_list=[
    ('pipeline_num', pipeline_num),
    ('pipeline_str', pipeline_str),
])


# In[15]:


#  パイプラプラインインを実行する
data_prepared = pipeline_main.fit_transform(data_train)
data_prepared


# In[16]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[17]:


# モデルの訓練
tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_label)
# 訓練セットの評価
data_predictions = tree_reg.predict(data_prepared)
tree_mse  = mean_squared_error(data_label, data_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

