
# coding: utf-8

# In[3]:


import os
import tarfile
from six.moves import urllib


# In[12]:


DOWNLAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL  = DOWNLAD_ROOT + 'datasets/housing/housing.tgz'


# In[15]:


tgz_path = os.path.join(HOUSING_PATH, 'housing.tgz')


# ### ディレクトリが存在しない場合に作成する

# In[19]:


if not os.path.isdir(HOUSING_PATH):
    os.makedirs(HOUSING_PATH)


# ### .tgzファイルをローカルに保存する

# In[22]:


print(HOUSING_URL)
print(tgz_path)
urllib.request.urlretrieve(HOUSING_URL, tgz_path)


# ### .tgzファイルを操作する
# * open：ファイルオープン
# * extractall：ファイルを展開する
# * close：クローズ

# In[25]:


housing_tgz = tarfile.open(tgz_path)
housing_tgz.extractall(path=HOUSING_PATH)
housing_tgz.close()

