
# coding: utf-8

# # SVM回帰の性能調査

# In[1]:


import myutil
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.svm import SVR


# In[2]:


df = myutil.DataFactory()


# # 回帰木の学習
# * 回帰木とは？
#  * 属性、属性値のカテゴリごとにツリーを生成
#  * 枝ごとに値を推測する
#  * https://mathwords.net/ketteigi

# In[3]:


data_prepared, data_label = df.getPreparedDataSet()
# モデルの訓練
tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_label)
# 訓練セットの評価
data_predictions = tree_reg.predict(data_prepared)
tree_mse  = mean_squared_error(data_label, data_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


# # SVMの学習
# * サポートベクトルマシンの略
# * 特徴
#  * 局所解収束がない
#  * データを２つに分類する問題に優れている
#   * 最適な分類境界線を探してくれる
#   * 多クラスの分類は苦手

# In[4]:


data_prepared, data_label = df.getPreparedDataSet()
# モデルの選択
svc = svm.SVC()
# 学習
svc.fit(data_prepared, data_label)


# In[ ]:


# 訓練セットの評価
data_predictions_svc = svc.predict(data_prepared)
svc_mse  = mean_squared_error(data_label, data_predictions_svc)
svc_rmse = np.sqrt(svc_mse)
print(svc_rmse)


# # SVM最良モデルの探索

# In[3]:


from sklearn.model_selection import GridSearchCV
data_prepared, data_label = df.getPreparedDataSet()
param_grid = [
        {'kernel': ['linear'], 'C': [10.,100., 1000., 10000.]},
        {'kernel': ['rbf'], 'C': [1.0, 10., 100., 1000.0],
         'gamma': [0.01, 0.1, 1.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search.fit(data_prepared, data_label)


# In[5]:


negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse


# In[6]:


grid_search.best_params_

