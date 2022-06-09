cfg = {
    "ensemble_size": 30,
    "num_rounds": 30,  
    "encCachePath": 'data/Checkpoint.pkl', 
    "preprocessCols": ['nl', 'country'],
    
    "param": {
        "max_depth": 20, 
        "eta": 0.03,   
        "gamma": 0.2,  
        "alpha": 0,
        "subsample": 1, 
        "min_child_weight": 3,
        "train_ratio": 1,
        
        "seed": 0,
        "verbosity": 0,
        "nthread": 7,
        "eval_metric": ["error", "logloss"],
        "objective": "binary:logistic",
    },
}


# ==============================================================================
# IMPORTS
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
from sklearn.preprocessing import OneHotEncoder

# ==============================================================================
# LOAD DATA
# ==============================================================================

csv_path = '../public_tables/'
image_path = sys.argv[1] + '/anon_images/'

train_csv_path = os.path.join(csv_path, 'train.csv')
train_df = pd.read_csv(train_csv_path, index_col = 0)


csv_rows = train_df.loc[:, ['filename', 'label']].to_dict(orient='records')
label_map = {x['filename']: x['label'] for x in csv_rows}

test_csv_path = os.path.join(csv_path, 'random_test_reduct.csv')
test1_df = pd.read_csv(test_csv_path, index_col = 0)

test_csv_path = os.path.join(csv_path, 'country_test_reduct.csv')
test2_df = pd.read_csv(test_csv_path, index_col = 0)


# ==============================================================================
# XGB Functions
# ==============================================================================

from lib.XGBoost_params import *

def bootstrap_sample(data):
    l = data.shape[0]
    C = choice(array(range(l)),l,replace=True)
    sample = data[C,:]
    return sample
    
def train_xgboost(dTrain, param, num_round):
    evallist = [(dTrain, 'train')]
    param = param_D2L(param)
    evals_result={}
    
    bst = xgb.train(param, 
                  dTrain, 
                  num_round,
                  evallist, 
                  verbose_eval=False, 
                  evals_result=evals_result)
    return bst, evals_result
    
def simple_bootstrap(Train, 
                 param, 
                 num_rounds = 10,
                 ensemble_size = 1, 
                 normalize=True):

    logs = []
    for i in trange(ensemble_size, leave=False):   
        boot_train = bootstrap_sample(Train)
        dTrain = to_DMatrix(boot_train)
        bst, results = train_xgboost(dTrain, param, num_round=num_rounds)
        logs.append({'i': i,'bst': bst,'results': results,})
    return logs

def getPred(bst, data):
    d = copy.deepcopy(data)
    d = to_DMatrix(d)
    d.label = None
    preds = bst.predict(d, output_margin=True)
    return preds
    
def process(preds, smean, sstd):
    preds = (preds-smean)/sstd

    _mean = np.mean(preds,axis=1)
    _std = np.std(preds,axis=1)

    pred_wo_abstention = (2*(_mean>0))-1
    pred_with_abstention = copy.deepcopy(pred_wo_abstention)
    pred_with_abstention[_std>abs(_mean)]=0
    return  pred_wo_abstention, pred_with_abstention

def evalMain(Data, labels, logs):
    preds = zeros([len(Data),len(logs)])
    for i in range(len(logs)):
        preds[:,i] = getPred(logs[i]['bst'], Data)

    smean = preds.mean()
    sstd = preds.std()
    pred1, pred2 = process(preds, smean, sstd)
    return (smean, sstd)

def getTestPreds(Data, logs, stats):
    preds = zeros([len(Data),len(logs)])
    for i in range(len(logs)):
        preds[:,i] = getPred(logs[i]['bst'], Data)

    smean, sstd = stats
    pred1, pred2 = process(preds, smean, sstd)
    return [pred1, pred2]


def plot_results(logs):
    plt.figure(figsize=[16,3])
    plt.subplot(1, 2, 1)
    plt.plot(logs[0]['results']['train']['error'], '.-', label='error')
    plt.grid(); plt.legend(prop={'size': 16});

    plt.subplot(1, 2, 2)
    plt.plot(logs[0]['results']['train']['logloss'], '.-', label='logloss')
    plt.grid(); plt.legend(prop={'size': 16})
    plt.show()

def loadEnc(encCachePath):
    encCache = pkl.load(open(encCachePath,'rb'))
    Train = encCache['train_enc'].data
    Test1 = encCache['test1_enc'].data
    Test2 = encCache['test2_enc'].data
    print(Train.shape, Test1.shape, Test2.shape)
    return Train, Test1, Test2

def preprocess(Data, df, cols=['nl', 'country']):
    if 'country' in cols:
        enc = OneHotEncoder(handle_unknown='ignore')
        country = df['country'].values.reshape(-1,1)
        countryOHE = enc.fit_transform(country).toarray()
        Data = np.hstack([countryOHE, Data])
        
    if 'nl' in cols:
        nl = df.nl_mean.values.reshape(-1,1)
        Data = np.hstack([nl, Data])
    
    return Data

def createDict(l):
    return {i:eval(i) for i in l}

def getUrbanRural(Data, df, labels=None, isTest=False):
    isUrban = df['urban'] == True
    uData = Data[isUrban]
    rData = Data[~isUrban]
    if not isTest:
        ulabels = labels[isUrban]
        rlabels = labels[~isUrban]
        return uData, rData, ulabels, rlabels
    else:
        return uData, rData

def getIdx(df):
    df = copy.deepcopy(df)
    df = df.reset_index(drop=True)
    uIdx = df[df['urban'] == True].index.values
    rIdx = df[df['urban'] == False].index.values
    return uIdx, rIdx
    
def getPredictions(df, uPred, rPred):
    uIdx, rIdx = getIdx(df)
    labels = np.ones(len(df)) * 10
    for i, idx in enumerate(uIdx):
        labels[idx] = uPred[i]
    for i, idx in enumerate(rIdx):
        labels[idx] = rPred[i]
    return labels

# ==============================================================================
# XGB Expts
# ==============================================================================


Train, Test1, Test2 = loadEnc(cfg['encCachePath'])
Train = preprocess(Train, train_df, cfg['preprocessCols'])
Test1 = preprocess(Test1, test1_df, cfg['preprocessCols'])
Test2 = preprocess(Test2, test2_df, cfg['preprocessCols'])

uTrain, rTrain, uTrain_labels, rTrain_labels = getUrbanRural(Train, train_df, Train[:,-1])
uTest1, rTest1 = getUrbanRural(Test1, test1_df, isTest=True)
uTest2, rTest2 = getUrbanRural(Test2, test2_df, isTest=True)


ulogs = simple_bootstrap(uTrain, cfg['param'], cfg['num_rounds'], cfg['ensemble_size'])
stats = evalMain(uTrain, uTrain_labels, ulogs)
utest1Metrics = getTestPreds(uTest1, ulogs, stats)
utest2Metrics = getTestPreds(uTest2, ulogs, stats)

rlogs = simple_bootstrap(rTrain, cfg['param'], cfg['num_rounds'], cfg['ensemble_size'])
stats = evalMain(rTrain, rTrain_labels, rlogs)
rtest1Metrics = getTestPreds(rTest1, rlogs, stats)
rtest2Metrics = getTestPreds(rTest2, rlogs, stats)

pred1_test1 = getPredictions(test1_df, utest1Metrics[0], rtest1Metrics[0])
pred2_test1 = getPredictions(test1_df, utest1Metrics[1], rtest1Metrics[1])

pred1_test2 = getPredictions(test2_df, utest2Metrics[0], rtest2Metrics[0])
pred2_test2 = getPredictions(test2_df, utest2Metrics[1], rtest2Metrics[1])


def genPredCSV(base_df, pred1, pred2):
    pred_df = copy.deepcopy(base_df)
    if 'dummy_label' in base_df.columns:
        pred_df = pred_df.drop(columns=['dummy_label'])
    pred_df = pred_df.drop(columns=['country', 'nl_mean'])
    
    pred_df['pred_with_abstention'] = pred2.astype(int)
    pred_df['pred_wo_abstention'] = pred1.astype(int)
    return pred_df

preds_test1 = genPredCSV(test1_df, pred1_test1, pred2_test1)
preds_test1.to_csv('data/results.csv', index=False)

preds_test2 = genPredCSV(test2_df, pred1_test2, pred2_test2)
preds_test2.to_csv('data/results_country.csv', index=False)
