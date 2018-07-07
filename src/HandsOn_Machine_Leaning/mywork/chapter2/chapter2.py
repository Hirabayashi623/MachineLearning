
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib


# In[2]:


DOWNLAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL  = DOWNLAD_ROOT + 'datasets/housing/housing.tgz'


# ## ネットワーク上の圧縮ファイルをローカルに保存する

# In[5]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[6]:


fetch_housing_data()


# ## CSVファイルを読込む（配列として扱える）

# In[1]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


# ## データ構造を確認する

# In[6]:


housing = load_housing_data()
housing.head()


# ## データの情報の確認
# * info()メソッドを使うことで以下のことが確認できる
#     * 総行数
#     * 各属性のタイプ
#     * Nullではない数

# In[7]:


#housing.info


# * 連想配列のように行を選択可能
# * value_counts関数によって個数の集計をとれる

# In[8]:


housing['ocean_proximity'].value_counts()


# In[9]:


import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
plt.show()


# # テストセットを作る
# * テストセットは全体の２０％ほどを確保する

# In[10]:


import numpy as np
def split_train_test(data, test_ratio=0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices  = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]


# In[11]:


train_set, test_set = split_train_test(housing)
print('trains:', len(train_set), ', tests: ', len(test_set))


# ## scikit_learnのデータ分割関数を利用する
# * 乱数生成時の初期化ができる（毎回同じ乱数が生成される）
# * 同じ分類のデータには同じ行番号を割り振る（データセット分割時に離れ離れにならない）

# In[12]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print('trains:', len(train_set), ', tests: ', len(test_set))


# In[13]:


train_set.hist(bins=50, figsize=(20,15))
plt.show()


# ## 収入カテゴリを作成する
# * 収入を連続的な値ではなく「〇〇～△△の間にどれだけいるか」という値に変換する
# * 注意点！
#  * 各層が十分に大きな値である必要がある
#  * 層の数が多くなりすぎおないようにする

# In[14]:


housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)             # 新しい列を追加する＆層の数を減らす
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)  # 値の小さい層を集約する
housing['income_cat'].hist()
plt.show()


# ## 層化抽出
# ### 層化抽出とは？
#     ある一定の割合を保ちながらサンプルを抽出すること

# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[16]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]


# In[17]:


# 分割前のデータ割合
print(housing['income_cat'].value_counts() / len(housing) * 100)
# 分割後のデータ割合
print(strat_train_set['income_cat'].value_counts() / len(strat_train_set) * 100)


# 分割用の列「income_cat」を削除　⇒　元のデータ構造に戻す

# In[18]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


# In[19]:


print(strat_train_set.columns.values)


# # 洞察を得るためにデータを研究、可視化する
# 注意点！
#  * 元のデータではなく、訓練データを用いる
#  * 訓練データを壊さないように、コピーしてから使用する

# In[20]:


housing = strat_train_set.copy()


# ## 地理データの可視化

# In[21]:


# 区域のシンプルな散布図
housing.plot(kind='scatter', x='longitude', y='latitude')
plt.show()


# ### 密度を表す
# * alpha引数を利用する
#  * alphaで透明度を指定
#  * 複数の点が重なると色が濃くなる　⇒　密度を表現できる

# In[22]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()


# ### 点の大きさによって人口を表現する

# In[23]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3,
            s=housing['population']/80,         # 円の半径を指定する
            label='population')
plt.show()


# ### 色によって住宅価格を表現する
# カラーマップ表はここ：https://matplotlib.org/examples/color/colormaps_reference.html

# In[24]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.3,
            s=housing['population']/80,         # 円の半径を指定する
            label='population',
            c='median_house_value',             # 住宅価格によって色を決定する
            cmap=plt.get_cmap('jet'),           # カラーマップの設定
            colorbar=True)
plt.show()


# ## 相関を探す

# In[25]:


# median_house_valueとの相関
print(housing.corr()['median_house_value'].sort_values(ascending=False))


# In[26]:


from pandas.plotting import scatter_matrix
# 上位４カラムに絞る
attribute = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attribute], figsize=(12,8))
plt.show()


# ### 相関が最も強い2データの散布図
# * 50万ドル近辺に横線が表れている
# * ⇒これはかかくの上限を50万ドルに設定したことによるもの
# * ⇒アルゴリズムが誤学習しないように取り除く必要がある

# In[27]:


housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
plt.show()


# ### 属性の組み合わせを試す
# * total_rooms   ：部屋数の合計　⇒　世帯あたりの値が知りたい
# * total_bedrooms：寝室数の合計　⇒　部屋あたりの値が知りたい
# * population    ：人口　　　　　⇒　世帯あたりの値も面白そう

# In[28]:


housing['room_per_household']       = housing['total_rooms']    / housing['households']
housing['bedrooms_per_room']        = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population']     / housing['households']


# In[29]:


# median_house_valueとの相関
print(housing.corr()['median_house_value'].sort_values(ascending=False))


# In[30]:


attribute = ['median_house_value', 'room_per_household', 'bedrooms_per_room', 'population_per_household']
scatter_matrix(housing[attribute], figsize=(12,8))
plt.show()


# ## 機械学習アルゴリズムに渡せるようにデータを準備する
# ### 予測子とラベルを分ける（ここでいう住宅価格とそれ以外の項目）
# * 必ずしも予測子とラベルにまったく同じ変換をかけるとは限らないため

# In[31]:


# dropは、特定キーを落としたデータを返す関数　※元のデータには影響を与えない
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# ### データをクリーニングする
# * ほとんどの機械学習アルゴリズムでは欠損特微量を処理でない（ここでいうtotal_bedrooms）
# * 対処方法は以下の3通り
#     1. 対応する区域を取り除く
#     2. 属性全体を取り除く
#     3. 何らかの値を設定する(0、平均値、中央値)

# In[32]:


# 対応する区域を取り除く
housing.dropna(subset=['total_rooms'])
# 属性全体を取り除く
housing.drop('total_rooms', axis=1)
# 何らかの値を設定する(0、平均値、中央値)
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=False)
print(median)


# ### Imputerの利用
# * scikit-learnが提供している欠損値を処理してくれるクラス
# * 数値属性でなければ計算できない　⇒　数値以外を取り除いたデータコピーを使用する
# * 本番稼働度に追加されるデータに欠損値がないとは限らないため、すべての数値属性に対して適応させる
# * 使い方
#     1. Imputerオブジェクト生成（欠損値の補い方を指定※ここでは中央値）
#     2. fit関数を用いてデータを渡す（この段階で欠損値を補う用の値が生成される）
#     3. statistics_の値を確認（欠損値を補う値が格納されている）

# In[33]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
housing_num = housing.drop(['ocean_proximity'], axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)


# * 欠損値を補うよう変換
# * 結果はNumpy配列で渡される

# In[34]:


x = imputer.transform(housing_num)
housing_tr = pd.DataFrame(x, columns=housing_num.columns)


# ### テキスト属性の処理
# * 機械学習アルゴリズムでは数値属性の方が処理がしやすい
# * カテゴリーのように個数が決まったテキストであれば数値に変換する
# * pandas.DataFrame.factorize関数を使用する
#    * 戻り値：数値化後のデータ、数値とテキストの対応表

# In[35]:


housing_cat = housing['ocean_proximity']
housing_cat_encoded, housing_categories = housing_cat.factorize()
print('***  encoded  ***\n%s\n*** categories ***\n%s' % (housing_cat_encoded, housing_categories))


# ### ワンホットエンコーディング
# * 上記の実装だと、アルゴリズムが近接した値が離れた値よりも「近い」と誤認識
# * ⇒カテゴリごと1個のバイナリ属性を作る
# * sklearnのOneHotEncoder.fit_transform関数を使用する

# In[36]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
# reshape(-1,1)で配列（1行N列）をN行1列の行列に変換
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print('scipyの疎行列\n', housing_cat_1hot)
print('numpyの行列\n',   housing_cat_1hot.toarray())


# ### CategoricalEncoder
# * 「テキスト属性を整数カテゴリ属性へ変換」「整数カテゴリ属性をワンホットエンコーディング」を1度に実行できる
# * sklearnで未実装？？

# ## カスタム変換器
# * カスタムクリーンアップや特定の属性の結合などは独自の変換器を定義する必要がある
# * scikit-learnでは継承ではなく「タッグタイピング」に依拠
# * 「fit」「transform」「fit_transform」の３つのメソッドを実装すればよい
#  * 「transform」「fit_transform」についてはTransformerMixinを基底クラスに追加すればOK！
# ### 「タッグタイピング」とは？
# * 例えばJavaではintarfaceを用いた型の制限が可能
# * interfaceを実装していなくても、interfaceのすべてのメソッドが実装されていればintarfaceを満たす型として扱える
# * 由来：アヒルのように歩き、アヒルのように鳴くものはアヒルに違いない

# In[37]:


from sklearn.base import BaseEstimator, TransformerMixin

room_idx, bedrooms_idx, population_idx, household_idx = 3, 4, 5, 6

class CombineedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household      = X[:, room_idx]       / X[:, household_idx]
        population_per_household = X[:, population_idx] / X[:, household_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx] / X[:, room_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[38]:


attr_adder = CombineedAttributeAdder()
housing_extra_attribs = attr_adder.transform(housing.values)
# new_housing = pd.DataFrame(housing,extra_attribs, columns=[...])


# ## 特微量のスケーリング
# * すべての属性のスケール（範囲、最大最小の幅）を統一する必要がある
#  * 入力数値のスケールが大きく異なると性能を発揮できないため
# * 方法は以下の2通り
#     1. 最小最大スケーリング
#      * 正規化とも呼ばれる
#      * 値を0～1の範囲に収める方法
#      * 計算方法：(val - min) / (max - min)
#      * MinMaxScalerが用意されている
#     2. 標準化
#      * (val - ave) / 分散
#      * 上下限がないため一部のアルゴリズム（例えばニューラルネットワーク）で問題になる
#      * 一方、外れ値（一部の大きな／小さな値）の影響が小さくなる
#      * StandardScalerが用意されている

# ## 変換パイプライン
# * scikit-learnにはPipelineクラスが用意されている
# * このクラスを使用することで、データ変換のステップを正しく実行してくれる
# * パイプラインには変換器の名前／変換器を設定する
#  * 名前は任意でOK

# In[39]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[40]:


# パイプラインの実装
num_pipeline = Pipeline([
    ('imputer'      , Imputer(strategy='median')),
    ('attribs_adder', CombineedAttributeAdder()),
    ('std_scaler'   , StandardScaler()), 
])


# In[41]:


# パイプラインの実行
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num.shape)
print(housing_num_tr.shape)


# ### pandas.DataFrameから数値の属性のみを抽出する変換器

# In[42]:


from sklearn.base import BaseEstimator, TransformerMixin
from future_encoders import OneHotEncoder


# In[43]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[44]:


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector'     , DataFrameSelector(num_attribs)),   # 数値の属性を抽出する
    ('imputer'      , Imputer(strategy='median')),       # 欠損値を中央値で補う
    ('attribs_adder', CombineedAttributeAdder()),        # 複合させて新しい属性を生成する
    ('std_scaler'   , StandardScaler()),                 # 標準化する
])

cat_pipeline = Pipeline([
    ('selector'   , DataFrameSelector(cat_attribs)),     # テキスト属性を抽出する
    ('cat_encoder', OneHotEncoder()),                    # ワンホットエンコーディングを実施
])


# ### ２つのパイプラインを１つに統合する

# In[45]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


# In[46]:


housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)


# # モデルを選択して訓練する

# In[47]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[48]:


# 一部のデータを抽出して訓練結果を確認する
some_data          = housing.iloc[:5]
some_labels        = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print('Predictions : ', lin_reg.predict(some_data_prepared))
print('Labels      : ', list(some_labels))


# ### RMSEの測定
# * RMSEとは？
#  * Room Mean Square Error：平均二乗誤差
# * skleanの「mean_squared_error」関数を使用する

# In[49]:


from sklearn.metrics import mean_squared_error


# In[50]:


# 訓練データに対して予測値を求める
housing_predictions = lin_reg.predict(housing_prepared)
# 二乗誤差を求める
lin_mse  = mean_squared_error(housing_labels, housing_predictions)
# 平均二乗誤差を求める
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


# ### 結果の考察
# * 推測対象のmedian_housing_valuesは120,000～265,000が大半
# * それに対する誤差としては大きい
#  * ⇒満足できる水準ではない
#  * ⇒訓練データに過少適合している例
# * 原因は？
#  1. 特微量が予測ができるほどの情報を提供していない
#  2. モデルの性能が低い
# * 対処法は？
#  1. より強力なモデルを選ぶ
#  2. 訓練アルゴリズムによりよい特微量を与える
#  3. モデルの制約を緩める（正規化を標準化に変更する等）

# ### より複雑なモデルを試してみる

# In[51]:


from sklearn.tree import DecisionTreeRegressor


# In[52]:


# モデルの訓練
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
# 訓練セットの評価
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse  = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


# # 交差検証を使ったより良い評価
# * 決定木モデルを評価する方法
#  1. train_test_split関数を用いて訓練セットと検証セットに分割する
#  2. 交差検証
#    * 訓練セットをN個のフィールドに無造作に分割
#    * N-1個のフィールドで訓練、1個のフィールドで評価
#    * 評価用のフィールドを変えながらN回訓練→評価する

# In[53]:


from sklearn.model_selection import cross_val_score


# In[54]:


def display_scores(scores):
    print('Scores            :', scores)
    print('Mean              :', scores.mean())
    print('Standard deviation:', scores.std())


# In[55]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                        scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[56]:


display_scores(tree_rmse_scores)


# ### 線形回帰モデルでも検証

# In[57]:


scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                        scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores)


# In[58]:


display_scores(lin_rmse_scores)


# ### ランダムフォレストモデルでも検証
# * ランダムフォレストモデルとは？
#  * 特微量の無作為なサブセットを使って多数の決定木を訓練
#  * アンサンブル学習：ほかの多数のモデルを基礎としてモデルを構築すること

# In[59]:


from sklearn.ensemble import RandomForestRegressor


# In[60]:


forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                        scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)


# In[61]:


display_scores(forest_rmse_scores)


# ### 可視化してみる

# In[62]:


x = np.arange(0,len(lin_rmse_scores))
plt.plot(x, lin_rmse_scores)
plt.plot(x, tree_rmse_scores)
plt.plot(x, forest_rmse_scores)
plt.ylim([0,100000])
plt.show()


# # モデルを微調整する
# ## グリッドサーチ
# * sklearnのGridSearchCVにサーチさせる
# * どのハイパパラメータを操作するか、その値として何を試すかを指定
# * 指定された値から得られるすべての組み合わせを交差検証で評価してくれる

# In[2]:


from sklearn.model_selection import GridSearchCV


# In[3]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]


# In[4]:


forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)


# ### 最良の推定器を取得する

# In[71]:


grid_search.best_estimator_


# ### 評価スコアを確認する

# In[74]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ## ランダムサーチ
# * グリッドサーチは比較的少ない組み合わせのときには有効
# * 組み合わせが多い（探索空間が大きい）時には、ランダムサーチを使用する
# * クラスは「RandomizedSearchCV」で使用方法はグリッドサーチと同様

# In[81]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# In[82]:


param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}


# In[83]:


forest_reg = RandomForestRegressor()
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
print(rnd_search.best_params_)


# ## 最良モデルと誤差の分析
# * 最良モデルをよく調べると、問題について深い洞察が得られることがある
# * ここでは、正確な予測のためにここの属性の相対的な重要性の大小がどうなっているのかを示すことができる

# In[1]:


feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

