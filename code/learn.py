LOAD_LIM = None
ENCODE_LIM = None

# LOAD_LIM = 100
# ENCODE_LIM = 100

# ==============================================================================
# PRELIM
# ==============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
import torchmetrics.functional as metrics
import ntpath
import collections

import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import pickle as pkl
import copy
import pickle
import sys

csv_path = '../public_tables/'
image_path = sys.argv[1] + '/anon_images/'
train_val_split_ratio = 0.9
batch_size = 2

train_csv_path = os.path.join(csv_path, 'train.csv')
train_df = pd.read_csv(train_csv_path, index_col = 0)

def getData(path, isTest=False):
    image = np.load(path)
    image = image.f.x
    if isTest:
        return image
    label = label_map[path.split('/')[-1]]
    return image, label

def loadTrain(train_paths, lim=None):
    X_train = []
    y_train = []

    for p in tqdm(train_paths[:lim]):
        x,y = getData(p)
        X_train.append(x)
        y_train.append(y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

def loadTest(train_paths, lim=None):
    X_test = []
    for p in tqdm(train_paths[:lim]):
        x = getData(p, isTest=True)
        X_test.append(x)

    X_test = np.array(X_test)
    return X_test

csv_rows = train_df.loc[:, ['filename', 'label']].to_dict(orient='records')
label_map = {x['filename']: x['label'] for x in csv_rows}
train_paths = [os.path.join(image_path, r['filename']) for r in csv_rows]

test_csv_path = os.path.join(csv_path, 'random_test_reduct.csv')
test1_df = pd.read_csv(test_csv_path, index_col = 0)
test1_paths = [os.path.join(image_path, row['filename']) for index,row in test1_df.iterrows()]

test_csv_path = os.path.join(csv_path, 'country_test_reduct.csv')
test2_df = pd.read_csv(test_csv_path, index_col = 0)
test2_paths = [os.path.join(image_path, row['filename']) for index,row in test2_df.iterrows()]

X_train, y_train = loadTrain(train_paths, lim=LOAD_LIM)
X_test1 = loadTest(test1_paths, lim=LOAD_LIM)
X_test2 = loadTest(test2_paths, lim=LOAD_LIM)

print(X_train.shape, y_train.shape)
print(X_test1.shape)
print(X_test2.shape)
print(train_df.shape, test1_df.shape, test2_df.shape)

# ==============================================================================
# KD Trees Encoding
# ==============================================================================

import os
import xgboost as xgb
from lib.XGBHelper import *
from lib.XGBoost_params import *
from lib.score_analysis import *
from lib.KDTreeEncoding import *
from lib.logger import logger

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from glob import glob
import pandas as pd
import pickle as pkl
import copy

def encode_df(df, tree, label_col, depth, lim=None, st=0):
    if lim == None:
        lim = len(df)
    df = copy.deepcopy(df[st:st+lim])
    df.index=df['filename']
    raw_data = encoded_dataset(image_path, df, tree, label_col=label_col, depth=depth)
    return raw_data

files = []
files += glob(f'{image_path}/*.npz')
print(len(files))

# encoderCfg = {
#     'max_images': 5000,
#     'tree_depth': 12
# }

encoderCfg = {
    'max_images': 10000,
    'tree_depth': 8
}

# encoderCfg = {
#     'max_images': 500,
#     'tree_depth': 8
# }

tree = train_encoder(files, max_images=encoderCfg['max_images'], tree_depth=encoderCfg['tree_depth'])[1]

train_enc = encode_df(train_df[:ENCODE_LIM], tree, depth=encoderCfg['tree_depth'], lim=None, label_col='label')

test1_df['dummy_label'] = 0
test1_enc = encode_df(test1_df[:ENCODE_LIM], tree, depth=encoderCfg['tree_depth'], lim=None, label_col='dummy_label')

test2_df['dummy_label'] = 0
test2_enc = encode_df(test2_df[:ENCODE_LIM], tree, depth=encoderCfg['tree_depth'], lim=None, label_col='dummy_label')

encCache = {
#     'tree': tree,
    'encoderCfg': encoderCfg,
    'train_enc': train_enc,
    'test1_enc': test1_enc,
    'test2_enc': test2_enc,
}

# filePath = 'data/encCache_maxImages=10000_treeDepth=8.pkl'
filePath = 'data/Checkpoint.pkl'


print('Saving Final cache file at : {}'.format(filePath))
pickle.dump(encCache, open(filePath, "wb"))















