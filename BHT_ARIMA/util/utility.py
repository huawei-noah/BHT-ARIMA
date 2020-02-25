#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle as pkl
import pandas as pd
import numpy as np
import time
import os

def load_data(dataset):
    filename = '../input/{}.npy'.format(dataset)
    data = np.load(filename, allow_pickle=True)
    return data

def compute_rmse(dataA, dataB):
    length = len(dataA)
    rmse = np.sqrt(np.sum([(a - b)**2 for a, b in zip(dataA, dataB)])/length)
    return rmse

def compute_rmse2(dataA, dataB):
    """ RMSE """
    t1 = np.sum((dataA - dataB) **2) / np.size(dataB)
    return np.sqrt(t1)

def iter_list(item, nums):
    return iter([item for _ in range(nums)])

def get_acc2(data1:np.ndarray, data2:np.ndarray)->float:
    acc_list = []
    for a, b in zip(data1, data2):
        if a < 0:
            acc_list.append(0)
        elif max(a, b)==0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)

def get_acc(y_pred, y_true):
    acc_list = []
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if a < 0:
            acc_list.append(0)
        elif max(a, b)==0:
            pass
        else:
            acc_list.append(min(a, b) / max(a, b))
    return sum(acc_list) / len(acc_list)

def generate_header(params_dict:dict)->str:
    header1 = "======== Configuration ========\n"
    header2 = ''
    for key in sorted(params_dict.keys(), key=len):
        header2 += "{} : {}\n".format(key,params_dict[key])
    
    header3="===============================\n"
    header = header1 + header2 + header3
    return header

def nd(y_pred, y_true):
    """ Normalized deviation"""
    t1 = np.sum(abs(y_pred-y_true)) / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return t1 / t2

def SMAPE(y_pred, y_true):
    s = 0
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    for a, b in zip(y_p, y_t):
        if abs(a) + abs(b) == 0:
            s += 0
        else:
            s += 2 * abs(a-b) / (abs(a) + abs(b))
    return s / np.size(y_true)

def nrmse(y_pred, y_true):
    """ Normalized RMSE"""
    t1 = np.linalg.norm(y_pred - y_true)**2 / np.size(y_true)
    t2 = np.sum(abs(y_true)) / np.size(y_true)
    return np.sqrt(t1) / t2

def get_index(y_pred, y_true):
    
    index_d = {}
    index_d['acc'] = get_acc(y_pred, y_true)
    index_d['rmse'] = compute_rmse2(y_pred, y_true)
    index_d['nrmse'] = nrmse(y_pred, y_true)
    index_d['nd'] = nd(y_pred, y_true)
    index_d['smape'] = SMAPE(y_pred, y_true)
    return index_d

def get_mean_index(index_list, key):
    return np.mean([index[key] for index in index_list])
    
def get_mean_index_dict(index_list):
    return { key:get_mean_index(index_list, key) for key in index_list[0].keys() }

def recorder(file:str, cfg:dict, per_d:dict):
    filename = file + '.csv'
    cols = ['dataset', 'Us_mode', 'p', 'd', 'q', 'taus', 'Rs', 'k', \
                'loop_time', 'testsize', 'acc', 'rmse','time', 'nrmse','nd', 'smape', 'info', 'run_date','log_file']
    
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.read_csv(filename)
    new_idx = df.index.size
    d = {}
    cfg.update(per_d)
    for key, val in cfg.items():
        if key in cols:
            d[key] = val        
    df = df.append(d, ignore_index=True)
    df.to_csv(filename, index=False)

def recorder_video(file:str, cfg:dict, per_d:dict):
    filename = file + '.csv'
    cols = ['dataset', 'Us_mode','mdt_mode', 'p', 'd', 'q', 'taus', 'Rs', 'k', \
                'loop_time', 'testsize', 'acc', 'rmse','time', 'nrmse','nd', 'smape', 'info', 'run_date','log_file']
    
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.read_csv(filename)
    new_idx = df.index.size
    d = {}
    cfg.update(per_d)
    for key, val in cfg.items():
        if key in cols:
            d[key] = val        
    df = df.append(d, ignore_index=True)
    df.to_csv(filename, index=False)


class CountTime(object):

    def __init__(self):
        self.start = time.time()
        pass
    
    def stop_timing(self):

        self.stop =  time.time()
    
    def mean_time(self, nums, mode=1):
        if mode == 1:
             return self.running_time(nums)
        elif mode == 2:
             return round((self.stop - self.start)/nums, 4)
    
    def running_time(self, nums=1):
        delta = (self.stop - self.start)/nums
        return "{}hours {}mins {}sec".format(int(delta//3600), int((delta%3600)//60), round(delta%60,4))
