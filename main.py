#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This script is the demo of BHT-ARIMA algorithm
# References : "Block Hankel Tensor ARIMA for Multiple Short Time Series Forecasting"


# import libraries
import numpy as np

from BHT_ARIMA import BHTARIMA
from BHT_ARIMA.util.utility import get_index

if __name__ == "__main__":
    # prepare data
    # the data should be arranged as (ITEM, TIME) pattern
    # import traffic dataset
    ori_ts = np.load('input/traffic_40.npy').T
    print("shape of data: {}".format(ori_ts.shape))
    print("This dataset have {} series, and each serie have {} time step".format(
        ori_ts.shape[0], ori_ts.shape[1]
    ))

    # parameters setting
    ts = ori_ts[..., :-1] # training data, 
    label = ori_ts[..., -1] # label, take the last time step as label
    p = 3 # p-order
    d = 2 # d-order
    q = 1 # q-order
    taus = [228, 5] # MDT-rank
    Rs = [5, 5] # tucker decomposition ranks
    k =  10 # iterations
    tol = 0.001 # stop criterion
    Us_mode = 4 # orthogonality mode

    # Run program
    # result's shape: (ITEM, TIME+1) ** only one step forecasting **
    model = BHTARIMA(ts, p, d, q, taus, Rs, k, tol, verbose=0, Us_mode=Us_mode)
    result, _ = model.run()
    pred = result[..., -1]

    # print extracted forecasting result and evaluation indexes
    print("forecast result(first 10 series):\n", pred[:10])

    print("Evaluation index: \n{}".format(get_index(pred, label)))



