import numpy as np
import os
import tarfile
from six.moves import urllib
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

# 定数の定義
DOWNLAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_NAME = 'housing.tgz'
HOUSING_URL  = DOWNLAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def downloadAfterCheck(housing_url=HOUSING_URL, housing_path=HOUSING_PATH, housing_name=HOUSING_NAME):
    if os.path.isfile(os.path.join(housing_path, housing_name)):
        None
    else:
        fetch_housing_data(housing_url, housing_path)

def load_housing_data(path):
    return pd.read_csv(path)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_columns):
        self.num_columns = num_columns
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.num_columns].values

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

class DataFactory():
    def __init__(self):
        self.data          = None
        self.data_train    = None
        self.data_test     = None
        self.data_prepared = None
        self.data_label    = None

    def split_dataset(self, data):
        # 層化分割用の属性を追加
        data['income_cat'] = np.ceil(data['median_income'] / 1.5)
        data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)

        # 層化分割
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(data, data['income_cat']):
            self.data_train = data.loc[train_index]
            self.data_test  = data.loc[test_index]
        self.data       = data

        # 層化分割用の属性を削除
        for set_ in (self.data_train, self.data_test):
            set_.drop('income_cat', axis=1, inplace=True)

        # 壊れないようにコピーを渡す
        return self.data_train.copy(), self.data_test.copy()

    def split_label(self, data, label_name):
        data_label = data[label_name].copy()
        data_other = data.drop(label_name, axis=1)

        return data_label, data_other

    def getPreparedDataSet(self):
        if self.data_prepared is None:
            # データのダウンロード
            downloadAfterCheck()

            # データの読込
            housing = load_housing_data('datasets/housing/housing.csv')

            df = DataFactory()
            # 訓練データとテストデータの分割
            data_train, data_test = df.split_dataset(housing)
            # ラベルとデータの分割
            data_label, data_train = df.split_label(data_train, 'median_house_value')

            # 数値属性について実施するパイプラインの定義
            str_columns = ['ocean_proximity']
            num_columns = list(data_train)
            for str_column in str_columns:
                num_columns.remove(str_column)

            pipeline_num = Pipeline([
                ('selector'  , DataFrameSelector(num_columns)),  # 数値属性だけ取り出す
                ('imputer'   , Imputer(strategy='median')),      # 欠損値を中央値で補う
                ('std_scalar', StandardScaler()),                # 標準化する
            ])

            pipeline_str = Pipeline([
                ('selector'   , DataFrameSelector(str_columns)), # 文字列属性だけ取り出す
                ('str_encoder', OneHotEncoder()),                # ワンホットエンコード
            ])

            # ２つのパイプラインで変換したデータを結合させるパイプライン
            pipeline_main = FeatureUnion(transformer_list=[
                ('pipeline_num', pipeline_num),
                ('pipeline_str', pipeline_str),
            ])

            #  パイプラプラインインを実行する
            self.data_prepared = pipeline_main.fit_transform(data_train)

            self.data_label = data_label

        else:
            None
        return self.data_prepared.copy(), self.data_label.copy()




